# -*- coding: utf-8 -*-
"""DQN distillation class for collect teacher's data and train student.

- Author: Kyunghwan Kim, Minseop Kim
- Contact: kh.kim@medipixel.io, minseop.kim@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
         https://arxiv.org/pdf/1511.06295.pdf (Policy Distillation)
"""

import os
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from rl_algorithms.common.buffer.distillation_buffer import DistillationBuffer
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.registry import AGENTS, build_learner


@AGENTS.register_module
class DistillationDQNAgent(DQNAgent):
    """
    DQN for policy distillation.

    - Use train function to train teacher and collect train phase data
    - Use _test function to collect teacher's distillation data.
    - Use update_distillation function to train student model.
    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.save_distillation_dir = None
        if not self.hyper_params.is_student:
            # Since raining teacher do not require DistillationBuffer,
            # it overloads DQNAgent._initialize.
            print("[INFO] Teacher mode.")
            DQNAgent._initialize(self)
            self.make_distillation_dir()
        else:
            # Training student or generating distillation data(test).
            print("[INFO] Student mode.")
            self.softmax_tau = 0.01

            build_args = dict(
                hyper_params=self.hyper_params,
                log_cfg=self.log_cfg,
                env_name=self.env_info.name,
                state_size=self.env_info.observation_space.shape,
                output_size=self.env_info.action_space.n,
                is_test=self.is_test,
                load_from=self.load_from,
            )
            self.learner = build_learner(self.learner_cfg, build_args)
            self.dataset_path = self.hyper_params.dataset_path

            self.memory = DistillationBuffer(
                self.hyper_params.batch_size, self.dataset_path
            )
            if self.is_test:
                self.make_distillation_dir()

    def make_distillation_dir(self):
        """Make directory for saving distillation data."""
        self.save_distillation_dir = os.path.join(
            self.hyper_params.save_dir,
            "distillation_buffer/" + self.env_info.name + "/" + self.log_cfg.curr_time,
        )
        os.makedirs(self.save_distillation_dir)
        self.save_count = 0

    def get_action_and_q(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state
        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        state = self._preprocess_state(state)
        q_values = self.learner.dqn(state)
        selected_action = q_values.argmax()
        selected_action = selected_action.detach().cpu().numpy()
        return selected_action, q_values.squeeze().detach().cpu().numpy()

    def step(
        self, action: np.ndarray, q_values: np.ndarray = None
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and store distillation data to buffer storage."""

        output = None
        if (
            self.is_test
            and hasattr(self, "memory")
            and not isinstance(self.memory, DistillationBuffer)
        ):
            # Teacher training's interim test.
            output = DQNAgent.step(self, action)
        else:
            current_ep_dir = f"{self.save_distillation_dir}/{self.save_count:07}.pkl"
            self.save_count += 1
            if self.is_test:
                # Generating expert's test-phase data.
                next_state, reward, done, info = self.env.step(action)
                with open(current_ep_dir, "wb") as f:
                    pickle.dump(
                        [self.curr_state, q_values], f, protocol=pickle.HIGHEST_PROTOCOL
                    )
                if self.save_count >= self.hyper_params.n_frame_from_last:
                    done = True
                output = next_state, reward, done, info
            else:
                # Teacher training.
                with open(current_ep_dir, "wb") as f:
                    pickle.dump([self.curr_state], f, protocol=pickle.HIGHEST_PROTOCOL)
                output = DQNAgent.step(self, action)

        return output

    def _test(self, interim_test: bool = False):
        """Test teacher and collect distillation data."""

        test_num = self.interim_test_num if interim_test else self.episode_num
        if hasattr(self, "memory"):
            # Teacher training interim test.
            score_list = []
            for i_episode in range(test_num):
                state = self.env.reset()
                done = False
                score = 0
                step = 0

                while not done:
                    if self.is_render:
                        self.env.render()

                    action = self.select_action(state)
                    next_state, reward, done, _ = self.step(action)

                    state = next_state
                    score += reward
                    step += 1

                print(
                    "[INFO] test %d\tstep: %d\ttotal score: %d"
                    % (i_episode, step, score)
                )
                score_list.append(score)

            if self.is_log:
                wandb.log(
                    {
                        "avg test score": round(sum(score_list) / len(score_list), 2),
                        "test total step": self.total_step,
                    }
                )
        else:
            # Gather test-phase data
            for i_episode in range(test_num):
                state = self.env.reset()
                done = False
                score = 0
                step = 0

                while not done:
                    if self.is_render:
                        self.env.render()

                    action, q_value = self.get_action_and_q(state)
                    next_state, reward, done, _ = self.step(action, q_value)

                    state = next_state
                    score += reward
                    step += 1

                print(
                    "[INFO] test %d\tstep: %d\ttotal score: %d\tbuffer_size: %d"
                    % (i_episode, step, score, self.save_count)
                )

                if self.is_log:
                    wandb.log({"test score": score})

                if self.save_count >= self.hyper_params.n_frame_from_last:
                    print(
                        "[INFO] test data saved completely. (%s)"
                        % (self.save_distillation_dir)
                    )
                    break

    def update_distillation(self) -> Tuple[torch.Tensor, ...]:
        """Make relaxed softmax target and KL-Div loss and updates student model's params."""
        states, q_values = self.memory.sample_for_diltillation()

        states = states.float().to(self.learner.device)
        q_values = q_values.float().to(self.learner.device)

        if torch.cuda.is_available():
            states = states.cuda(non_blocking=True)
            q_values = q_values.cuda(non_blocking=True)

        pred_q = self.learner.dqn(states)
        target = F.softmax(q_values / self.softmax_tau, dim=1)
        log_softmax_pred_q = F.log_softmax(pred_q, dim=1)
        loss = F.kl_div(log_softmax_pred_q, target, reduction="sum")

        self.learner.dqn_optim.zero_grad()
        loss.backward()
        self.learner.dqn_optim.step()

        return loss.item(), pred_q.mean().item()

    def add_expert_q(self):
        """Generate Q of gathered states using laoded agent."""
        self.make_distillation_dir()
        file_name_list = []

        for _dir in self.hyper_params.dataset_path:
            data = os.listdir(_dir)
            file_name_list += ["./" + _dir + "/" + x for x in data]

        for i in tqdm(range(len(file_name_list))):
            with open(file_name_list[i], "rb") as f:
                state = pickle.load(f)[0]

            torch_state = numpy2floattensor(state, self.learner.device)
            pred_q = self.learner.dqn(torch_state).squeeze().detach().cpu().numpy()

            with open(self.save_distillation_dir + "/" + str(i) + ".pkl", "wb") as f:
                pickle.dump([state, pred_q], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"Data containing expert Q has been saved at {self.save_distillation_dir}"
        )

    def train(self):
        """Execute appropriate learning code according to the running type."""
        if self.hyper_params.is_student:
            self.memory.reset_dataloader()
            if not self.memory.is_contain_q:
                print("train-phase student training. Generating expert agent Q..")
                assert (
                    self.load_from is not None
                ), "Train-phase training requires expert agent. Please use load-from argument."
                self.add_expert_q()
                self.hyper_params.dataset_path = [self.save_distillation_dir]
                self.load_from = None
                self._initialize()
                self.memory.reset_dataloader()
                print("start student training..")

            # train student
            assert self.memory.buffer_size >= self.hyper_params.batch_size
            if self.is_log:
                self.set_wandb()

            iter_1 = self.memory.buffer_size // self.hyper_params.batch_size
            train_steps = iter_1 * self.hyper_params.epochs
            print(
                f"[INFO] Total epochs: {self.hyper_params.epochs}\t Train steps: {train_steps}"
            )
            n_epoch = 0
            for steps in range(train_steps):
                loss = self.update_distillation()

                if self.is_log:
                    wandb.log({"dqn loss": loss[0], "avg q values": loss[1]})

                if steps % iter_1 == 0:
                    print(
                        f"Training {n_epoch} epochs, {steps} steps.. "
                        + f"loss: {loss[0]}, avg_q_value: {loss[1]}"
                    )
                    self.learner.save_params(steps)
                    n_epoch += 1
                    self.memory.reset_dataloader()

            self.learner.save_params(steps)

        else:
            DQNAgent.train(self)

            if self.hyper_params.n_frame_from_last is not None:
                # Copy last n_frame in new directory.
                print("saving last %d frames.." % self.hyper_params.n_frame_from_last)
                last_frame_dir = (
                    self.save_distillation_dir
                    + "_last_%d/" % self.hyper_params.n_frame_from_last
                )
                os.makedirs(last_frame_dir)

                # Load directory.
                episode_dir_list = sorted(os.listdir(self.save_distillation_dir))
                episode_dir_list = episode_dir_list[
                    -self.hyper_params.n_frame_from_last :
                ]
                for _dir in tqdm(episode_dir_list):
                    with open(self.save_distillation_dir + "/" + _dir, "rb") as f:
                        tmp = pickle.load(f)
                    with open(last_frame_dir + _dir, "wb") as f:
                        pickle.dump(tmp, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("\nsuccessfully saved")
                print(f"All train-phase dir: {self.save_distillation_dir}/")
                print(
                    f"last {self.hyper_params.n_frame_from_last} frames dir: {last_frame_dir}"
                )
