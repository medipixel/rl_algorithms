# -*- coding: utf-8 -*-
"""SAC agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
"""

import argparse
import time
from typing import Tuple

import gym
import numpy as np
from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.mlp import MLP, FlattenMLP, TanhGaussianDistParams
from rl_algorithms.registry import AGENTS
from rl_algorithms.utils.config import ConfigDict
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class SACAgent(Agent):
    """SAC agent interacting with environment.

    Attrtibutes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        actor_optim (Optimizer): optimizer for training actor
        critic_1 (nn.Module): critic model to predict state values
        critic_2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optim1 (Optimizer): optimizer for training critic_1
        critic_optim2 (Optimizer): optimizer for training critic_2
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        update_step (int): step number of updates
        i_episode (int): current episode number
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optim (Optimizer): optimizer for alpha

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
    ):
        """Initialize.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0
        self.update_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # target entropy
        target_entropy = -np.prod((self.action_dim,)).item()  # heuristic
        # automatic entropy tuning
        if hyper_params.auto_entropy_tuning:
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=optim_cfg.lr_entropy)

        self._initialize()
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        # create actor
        self.actor = TanhGaussianDistParams(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.network_cfg.hidden_sizes_actor,
        ).to(device)

        # create v_critic
        self.vf = MLP(
            input_size=self.state_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_vf,
        ).to(device)
        self.vf_target = MLP(
            input_size=self.state_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_vf,
        ).to(device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        # create q_critic
        self.qf_1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_qf,
        ).to(device)
        self.qf_2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_qf,
        ).to(device)

        # create optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )
        self.vf_optim = optim.Adam(
            self.vf.parameters(),
            lr=self.optim_cfg.lr_vf,
            weight_decay=self.optim_cfg.weight_decay,
        )
        self.qf_1_optim = optim.Adam(
            self.qf_1.parameters(),
            lr=self.optim_cfg.lr_qf1,
            weight_decay=self.optim_cfg.weight_decay,
        )
        self.qf_2_optim = optim.Adam(
            self.qf_2.parameters(),
            lr=self.optim_cfg.lr_qf2,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)

        # if initial random action should be conducted
        if (
            self.total_step < self.hyper_params.initial_random_action
            and not self.args.test
        ):
            return np.array(self.env.action_space.sample())

        if self.args.test and not self.is_discrete:
            _, _, _, selected_action, _ = self.actor(state)
        else:
            selected_action, _, _, _, _ = self.actor(state)

        return selected_action.detach().cpu().numpy()

    # pylint: disable=no-self-use
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = torch.FloatTensor(state).to(device)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        self.memory.add(transition)

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        self.update_step += 1

        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)

        # train alpha
        if self.hyper_params.auto_entropy_tuning:
            alpha_loss = (
                -self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params.w_entropy

        # Q function loss
        masks = 1 - dones
        q_1_pred = self.qf_1(states, actions)
        q_2_pred = self.qf_2(states, actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.hyper_params.gamma * v_target * masks
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # V function loss
        v_pred = self.vf(states)
        q_pred = torch.min(
            self.qf_1(states, new_actions), self.qf_2(states, new_actions)
        )
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        # train Q functions
        self.qf_1_optim.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optim.step()

        self.qf_2_optim.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optim.step()

        # train V function
        self.vf_optim.zero_grad()
        vf_loss.backward()
        self.vf_optim.step()

        if self.update_step % self.hyper_params.policy_update_freq == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            # regularization
            if not self.is_discrete:  # iff the action is continuous
                mean_reg = self.hyper_params.w_mean_reg * mu.pow(2).mean()
                std_reg = self.hyper_params.w_std_reg * std.pow(2).mean()
                pre_activation_reg = self.hyper_params.w_pre_activation_reg * (
                    pre_tanh_value.pow(2).sum(dim=-1).mean()
                )
                actor_reg = mean_reg + std_reg + pre_activation_reg

                # actor loss + regularization
                actor_loss += actor_reg

            # train actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            common_utils.soft_update(self.vf, self.vf_target, self.hyper_params.tau)
        else:
            actor_loss = torch.zeros(1)

        return (
            actor_loss.item(),
            qf_1_loss.item(),
            qf_2_loss.item(),
            vf_loss.item(),
            alpha_loss.item(),
        )

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor"])
        self.qf_1.load_state_dict(params["qf_1"])
        self.qf_2.load_state_dict(params["qf_2"])
        self.vf.load_state_dict(params["vf"])
        self.vf_target.load_state_dict(params["vf_target"])
        self.actor_optim.load_state_dict(params["actor_optim"])
        self.qf_1_optim.load_state_dict(params["qf_1_optim"])
        self.qf_2_optim.load_state_dict(params["qf_2_optim"])
        self.vf_optim.load_state_dict(params["vf_optim"])

        if self.hyper_params.auto_entropy_tuning:
            self.alpha_optim.load_state_dict(params["alpha_optim"])

        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):  # type: ignore
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "qf_1": self.qf_1.state_dict(),
            "qf_2": self.qf_2.state_dict(),
            "vf": self.vf.state_dict(),
            "vf_target": self.vf_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "qf_1_optim": self.qf_1_optim.state_dict(),
            "qf_2_optim": self.qf_2_optim.state_dict(),
            "vf_optim": self.vf_optim.state_dict(),
        }

        if self.hyper_params.auto_entropy_tuning:
            params["alpha_optim"] = self.alpha_optim.state_dict()

        Agent.save_params(self, params, n_episode)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, policy_update_freq, avg_time_cost = log_value
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "total loss: %.3f actor_loss: %.3f qf_1_loss: %.3f qf_2_loss: %.3f "
            "vf_loss: %.3f alpha_loss: %.3f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0] * policy_update_freq,  # actor loss
                loss[1],  # qf_1 loss
                loss[2],  # qf_2 loss
                loss[3],  # vf loss
                loss[4],  # alpha loss
                avg_time_cost,
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * policy_update_freq,
                    "qf_1 loss": loss[1],
                    "qf_2 loss": loss[2],
                    "vf loss": loss[3],
                    "alpha loss": loss[4],
                    "time per each step": avg_time_cost,
                }
            )

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.actor, self.vf, self.qf_1, self.qf_2], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                state = next_state
                score += reward

                # training
                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
                        loss = self.update_model()
                        loss_episode.append(loss)  # for logging

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                log_value = (
                    self.i_episode,
                    avg_loss,
                    score,
                    self.hyper_params.policy_update_freq,
                    avg_time_cost,
                )
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.save_params(self.i_episode)
        self.interim_test()
