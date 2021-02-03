import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from rl_algorithms.common.abstract.learner import TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.ddpg.learner import DDPGLearner
from rl_algorithms.registry import LEARNERS


@LEARNERS.register_module
class BCDDPGLearner(DDPGLearner):
    """Learner for BCDDPG Agent.

    Attributes:
        hyper_params (ConfigDict): hyper-parameters
        optim_cfg (ConfigDict): config of optimizer
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optim (Optimizer): optimizer for training actor
        critic_optim (Optimizer): optimizer for training critic

    """

    def update_model(
        self, experience: TensorTuple, demos: TensorTuple
    ) -> TensorTuple:  # type: ignore
        """Update actor and critic networks."""
        exp_states, exp_actions, exp_rewards, exp_next_states, exp_dones = experience
        demo_states, demo_actions, demo_rewards, demo_next_states, demo_dones = demos

        states = torch.cat((exp_states, demo_states), dim=0)
        actions = torch.cat((exp_actions, demo_actions), dim=0)
        rewards = torch.cat((exp_rewards, demo_rewards), dim=0)
        next_states = torch.cat((exp_next_states, demo_next_states), dim=0)
        dones = torch.cat((exp_dones, demo_dones), dim=0)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + (self.hyper_params.gamma * next_values * masks)
        curr_returns = curr_returns.to(self.device)

        # critic loss
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)

        # train critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # policy loss
        actions = self.actor(states)
        policy_loss = -self.critic(torch.cat((states, actions), dim=-1)).mean()

        # bc loss
        pred_actions = self.actor(demo_states)
        qf_mask = torch.gt(
            self.critic(torch.cat((demo_states, demo_actions), dim=-1)),
            self.critic(torch.cat((demo_states, pred_actions), dim=-1)),
        ).to(self.device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = torch.zeros(1, device=self.device)
        else:
            bc_loss = (
                torch.mul(pred_actions, qf_mask) - torch.mul(demo_actions, qf_mask)
            ).pow(2).sum() / n_qf_mask

        # train actor: pg loss + BC loss
        actor_loss = (
            self.hyper_params.lambda1 * policy_loss
            + self.hyper_params.lambda2 * bc_loss
        )
        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        common_utils.soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        common_utils.soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        return actor_loss.item(), critic_loss.item(), n_qf_mask
