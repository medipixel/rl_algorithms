from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from rl_algorithms.common.abstract.learner import TensorTuple
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.gail.networks import Discriminator
from rl_algorithms.ppo.learner import PPOLearner
import rl_algorithms.ppo.utils as ppo_utils
from rl_algorithms.registry import LEARNERS, build_backbone
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class GAILPPOLearner(PPOLearner):
    """PPO-based GAILLearner for GAIL Agent.

    Attributes:
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        actor (nn.Module): actor model to select actions
        critic (nn.Module): critic model to predict state values
        discriminator (nn.Module): discriminator model to classify data
        actor_optim (Optimizer): optimizer for training actor
        critic_optim (Optimizer): optimizer for training critic
        discriminator_optim (Optimizer): optimizer for training discriminator

    """

    def __init__(
        self,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
        env_name: str,
        state_size: tuple,
        output_size: int,
        is_test: bool,
        load_from: str,
    ):
        head.discriminator.configs.state_size = state_size
        head.discriminator.configs.action_size = output_size

        super().__init__(
            hyper_params,
            log_cfg,
            backbone,
            head,
            optim_cfg,
            env_name,
            state_size,
            output_size,
            is_test,
            load_from,
        )

        self.demo_memory = None

    def _init_network(self):
        """Initialize networks and optimizers."""
        # create actor
        if self.backbone_cfg.shared_actor_critic:
            shared_backbone = build_backbone(self.backbone_cfg.shared_actor_critic)
            self.actor = Brain(
                self.backbone_cfg.shared_actor_critic,
                self.head_cfg.actor,
                shared_backbone,
            )
            self.critic = Brain(
                self.backbone_cfg.shared_actor_critic,
                self.head_cfg.critic,
                shared_backbone,
            )
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)
        else:
            self.actor = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(
                self.device
            )
            self.critic = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(
                self.device
            )
        self.discriminator = Discriminator(
            self.backbone_cfg.discriminator,
            self.head_cfg.discriminator,
            self.head_cfg.aciton_embedder,
        ).to(self.device)

        # create optimizer
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=self.optim_cfg.lr_critic,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.discriminator_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=self.optim_cfg.lr_discriminator,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load model parameters
        if self.load_from is not None:
            self.load_params(self.load_from)

    def update_model(self, experience: TensorTuple, epsilon: float) -> TensorTuple:
        """Update generator(actor), critic and discriminator networks."""
        states, actions, rewards, values, log_probs, next_state, masks = experience
        next_state = numpy2floattensor(next_state, self.device)
        with torch.no_grad():
            next_value = self.critic(next_state)

        returns = ppo_utils.compute_gae(
            next_value,
            rewards,
            masks,
            values,
            self.hyper_params.gamma,
            self.hyper_params.tau,
        )

        states = torch.cat(states)
        actions = torch.cat(actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(values).detach()
        log_probs = torch.cat(log_probs).detach()
        advantages = (returns - values).detach()

        if self.hyper_params.standardize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        actor_losses, critic_losses, total_losses, discriminator_losses = [], [], [], []

        for (
            state,
            action,
            old_value,
            old_log_prob,
            return_,
            adv,
            epoch,
        ) in ppo_utils.ppo_iter(
            self.hyper_params.epoch,
            self.hyper_params.batch_size,
            states,
            actions,
            values,
            log_probs,
            returns,
            advantages,
        ):

            # critic_loss
            value = self.critic(state)
            if self.hyper_params.use_clipped_value_loss:
                value_pred_clipped = old_value + torch.clamp(
                    (value - old_value), -epsilon, epsilon
                )
                value_loss_clipped = (return_ - value_pred_clipped).pow(2)
                value_loss = (return_ - value).pow(2)
                critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()
            else:
                critic_loss = 0.5 * (return_ - value).pow(2).mean()
            critic_loss_ = self.hyper_params.w_value * critic_loss

            # train critic
            self.critic_optim.zero_grad()
            critic_loss_.backward()
            clip_grad_norm_(
                self.critic.parameters(), self.hyper_params.gradient_clip_cr
            )
            self.critic_optim.step()

            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            surr_loss = ratio * adv
            clipped_surr_loss = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
            actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean()

            # entropy
            entropy = dist.entropy().mean()
            actor_loss_ = actor_loss - self.hyper_params.w_entropy * entropy

            # train actor
            self.actor_optim.zero_grad()
            actor_loss_.backward()
            clip_grad_norm_(self.actor.parameters(), self.hyper_params.gradient_clip_ac)
            self.actor_optim.step()

            # total_loss
            total_loss = critic_loss_ + actor_loss_

            # discriminator loss
            demo_state, demo_action = self.demo_memory.sample(len(state))
            exp_score = torch.sigmoid(self.discriminator.forward((state, action)))
            demo_score = torch.sigmoid(
                self.discriminator.forward((demo_state, demo_action))
            )
            discriminator_exp_acc = (exp_score > 0.5).float().mean().item()
            discriminator_demo_acc = (demo_score <= 0.5).float().mean().item()
            discriminator_loss = F.binary_cross_entropy(
                exp_score, torch.ones_like(exp_score)
            ) + F.binary_cross_entropy(demo_score, torch.zeros_like(demo_score))

            # train discriminator
            if (
                discriminator_exp_acc < self.optim_cfg.discriminator_acc_threshold
                or discriminator_demo_acc < self.optim_cfg.discriminator_acc_threshold
                and epoch == 0
            ):
                self.discriminator_optim.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optim.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            total_losses.append(total_loss.item())
            discriminator_losses.append(discriminator_loss.item())

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        total_loss = sum(total_losses) / len(total_losses)
        discriminator_loss = sum(discriminator_losses) / len(discriminator_losses)

        return (
            (actor_loss, critic_loss, total_loss, discriminator_loss),
            (discriminator_exp_acc, discriminator_demo_acc),
        )

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
            "discriminator_optim_state_dict": self.discriminator_optim.state_dict(),
        }
        PPOLearner._save_params(self, params, n_episode)

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        PPOLearner.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def get_state_dict(self) -> Tuple[OrderedDict]:
        """Return state dicts, mainly for distributed worker."""
        return (
            self.actor.state_dict(),
            self.critic.state_dict(),
            self.discriminator.state_dict(),
        )

    def set_demo_memory(self, demo_memory):
        self.demo_memory = demo_memory
