import torch
import torch.nn.functional as F

from rl_algorithms.common.abstract.learner import TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.registry import LEARNERS
from rl_algorithms.sac.learner import SACLearner


@LEARNERS.register_module
class BCSACLearner(SACLearner):
    """Learner for BCSAC Agent.

    Attributes:
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
    """

    def update_model(
        self, experience: TensorTuple, demos: TensorTuple
    ) -> TensorTuple:  # type: ignore
        """Train the model after each episode."""
        self.update_step += 1

        states, actions, rewards, next_states, dones = experience
        demo_states, demo_actions, _, _, _ = demos
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)
        pred_actions, _, _, _, _ = self.actor(demo_states)

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
        states_actions = torch.cat((states, actions), dim=-1)
        q_1_pred = self.qf_1(states_actions)
        q_2_pred = self.qf_2(states_actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.hyper_params.gamma * v_target * masks
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # V function loss
        states_actions = torch.cat((states, new_actions), dim=-1)
        v_pred = self.vf(states)
        q_pred = torch.min(self.qf_1(states_actions), self.qf_2(states_actions))
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        # update actor
        actor_loss = torch.zeros(1)
        n_qf_mask = 0
        if self.update_step % self.hyper_params.policy_update_freq == 0:
            # bc loss
            qf_mask = torch.gt(
                self.qf_1(torch.cat((demo_states, demo_actions), dim=-1)),
                self.qf_1(torch.cat((demo_states, pred_actions), dim=-1)),
            ).to(self.device)
            qf_mask = qf_mask.float()
            n_qf_mask = int(qf_mask.sum().item())

            if n_qf_mask == 0:
                bc_loss = torch.zeros(1, device=self.device)
            else:
                bc_loss = (
                    torch.mul(pred_actions, qf_mask) - torch.mul(demo_actions, qf_mask)
                ).pow(2).sum() / n_qf_mask

            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()
            actor_loss = (
                self.hyper_params.lambda1 * actor_loss
                + self.hyper_params.lambda2 * bc_loss
            )

            # regularization
            mean_reg, std_reg = (
                self.hyper_params.w_mean_reg * mu.pow(2).mean(),
                self.hyper_params.w_std_reg * std.pow(2).mean(),
            )
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

        return (
            actor_loss.item(),
            qf_1_loss.item(),
            qf_2_loss.item(),
            vf_loss.item(),
            alpha_loss.item(),
            n_qf_mask,
        )
