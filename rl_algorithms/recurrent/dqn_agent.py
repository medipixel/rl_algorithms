"""R2D1 agent, which implement R2D2 but without distributed actor.

- Author: Euijin Jeong
- Contact: euijin.jeong@medipixel.io
- Paper: https://openreview.net/pdf?id=r1lyTjAqYX (R2D2)
"""

import time
from typing import Tuple

import cv2
import numpy as np
import torch
import wandb

from rl_algorithms.common.buffer.replay_buffer import RecurrentReplayBuffer
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.common.saliency_map import make_saliency_dir, save_saliency_maps
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.registry import AGENTS, build_learner


@AGENTS.register_module
class R2D1Agent(DQNAgent):
    """R2D1 interacting with environment.

    Attribute:
        memory (RecurrentPrioritizedReplayBuffer): replay memory for recurrent agent
        memory_n (RecurrentReplayBuffer): nstep replay memory for recurrent agent
    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.is_test:

            self.memory = RecurrentReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                self.hyper_params.sequence_size,
                self.hyper_params.overlap_size,
                n_step=self.hyper_params.n_step,
                gamma=self.hyper_params.gamma,
            )
            self.memory = PrioritizedBufferWrapper(
                self.memory, alpha=self.hyper_params.per_alpha
            )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = RecurrentReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    self.hyper_params.sequence_size,
                    self.hyper_params.overlap_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

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

    def select_action(
        self,
        state: np.ndarray,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: np.ndarray,
    ) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        state = self._preprocess_state(state)
        with torch.no_grad():
            selected_action, hidden_state = self.learner.dqn(
                state, hidden_state, prev_action, prev_reward
            )
        selected_action = selected_action.detach().argmax().cpu().numpy()

        if not self.is_test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        return selected_action, hidden_state

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        # Add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # Add a single step transition
        # If transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def step(
        self, action: np.ndarray, hidden_state: torch.Tensor
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)
        if not self.is_test:
            # if the last state is not a terminal state, store done as false
            done_bool = False if self.episode_step == self.max_episode_steps else done

            transition = (
                self.curr_state,
                action,
                hidden_state.detach(),
                reward,
                next_state,
                done_bool,
            )
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def sample_experience(self) -> Tuple[torch.Tensor, ...]:
        experiences_1 = self.memory.sample(self.per_beta)
        experiences_1 = (
            numpy2floattensor(experiences_1[:3], self.learner.device)
            + (experiences_1[3],)
            + numpy2floattensor(experiences_1[4:6], self.learner.device)
            + (experiences_1[6:])
        )
        if self.use_n_step:
            indices = experiences_1[-2]
            experiences_n = self.memory_n.sample(indices)
            return (
                experiences_1,
                numpy2floattensor(experiences_n[:3], self.learner.device)
                + (experiences_n[3],)
                + numpy2floattensor(experiences_n[4:], self.learner.device),
            )

        return experiences_1

    def train(self):
        """Train the agent."""
        # Logger
        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")

        # Pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.episode_num + 1):
            state = self.env.reset()
            hidden_in = torch.zeros(
                [1, 1, self.learner.gru_cfg.rnn_hidden_size], dtype=torch.float
            ).to(self.learner.device)
            prev_action = torch.zeros(
                1, 1, self.learner.head_cfg.configs.output_size
            ).to(self.learner.device)
            prev_reward = torch.zeros(1, 1, 1).to(self.learner.device)
            self.episode_step = 0
            self.sequence_step = 0
            losses = list()
            done = False
            score = 0

            t_begin = time.time()

            while not done:
                if self.is_render and self.i_episode >= self.render_after:
                    self.env.render()

                action, hidden_out = self.select_action(
                    state, hidden_in, prev_action, prev_reward
                )
                next_state, reward, done, _ = self.step(action, hidden_in)
                self.total_step += 1
                self.episode_step += 1

                if self.episode_step % self.hyper_params.sequence_size == 0:
                    self.sequence_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.sequence_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            info = self.learner.update_model(experience)
                            loss = info[0:2]
                            indices, new_priorities = info[2:4]
                            losses.append(loss)  # For logging
                            self.memory.update_priorities(indices, new_priorities)

                    # Decrease epsilon
                    self.epsilon = max(
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon,
                    )

                    # Increase priority beta
                    fraction = min(float(self.i_episode) / self.episode_num, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

                hidden_in = hidden_out
                state = next_state
                prev_action = common_utils.make_one_hot(
                    torch.as_tensor(action), self.learner.head_cfg.configs.output_size
                )
                prev_reward = torch.as_tensor(reward).to(self.learner.device)
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)

                if self.i_episode % self.save_period == 0:
                    self.learner.save_params(self.i_episode)
                    self.interim_test()

        # Termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.interim_test_num
        else:
            test_num = self.episode_num
        score_list = []
        for i_episode in range(test_num):
            hidden_in = torch.zeros(
                [1, 1, self.learner.gru_cfg.rnn_hidden_size], dtype=torch.float
            ).to(self.learner.device)
            prev_action = torch.zeros(
                1, 1, self.learner.head_cfg.configs.output_size
            ).to(self.learner.device)
            prev_reward = torch.zeros(1, 1, 1).to(self.learner.device)
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.is_render:
                    self.env.render()

                action, hidden_out = self.select_action(
                    state, hidden_in, prev_action, prev_reward
                )
                next_state, reward, done, _ = self.step(action, hidden_in)

                hidden_in = hidden_out
                state = next_state
                prev_action = common_utils.make_one_hot(
                    torch.as_tensor(action), self.learner.head_cfg.configs.output_size
                )
                prev_reward = torch.as_tensor(reward).to(self.learner.device)
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            score_list.append(score)

        if self.is_log:
            wandb.log(
                {
                    "avg test score": round(sum(score_list) / len(score_list), 2),
                    "test total step": self.total_step,
                }
            )

    def test_with_saliency_map(self):
        """Test agent with saliency map."""
        saliency_map_dir = make_saliency_dir(self.args.load_from.split("/")[-2])
        print(f"Save saliency map in directory : {saliency_map_dir}")
        print("Saving saliency maps...")
        i = 0
        for i_episode in range(self.args.episode_num):
            hidden_in = torch.zeros(
                [1, 1, self.learner.gru_cfg.rnn_hidden_size], dtype=torch.float
            ).to(self.learner.device)
            prev_action = torch.zeros(
                1, 1, self.learner.head_cfg.configs.output_size
            ).to(self.learner.device)
            prev_reward = torch.zeros(1, 1, 1).to(self.learner.device)
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            key = 0
            print("\nPress Any Key to move to next step... (quit: ESC key)")
            while not done:
                action, hidden_out = self.select_action(
                    state, hidden_in, prev_action, prev_reward
                )
                for param in self.learner.dqn.parameters():
                    param.requires_grad = False
                saliency_map = save_saliency_maps(
                    i,
                    (state, hidden_in, prev_action, prev_reward),
                    action,
                    self.learner.dqn,
                    self.learner.device,
                    saliency_map_dir,
                )
                i += 1
                next_state, reward, done, _ = self.step(action, hidden_in)

                state = np.transpose(state[-1])
                state = cv2.cvtColor(state, cv2.COLOR_GRAY2BGR)
                state = cv2.resize(state, (150, 150), interpolation=cv2.INTER_LINEAR)

                # Get Grad-CAM image
                result_images = None
                saliency_map = np.asarray(saliency_map)
                saliency_map = cv2.resize(
                    saliency_map, (150, 150), interpolation=cv2.INTER_LINEAR
                )
                saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_RGBA2BGR)
                overlay = cv2.addWeighted(state, 1.0, saliency_map, 0.5, 0)
                result = np.hstack([state, saliency_map, overlay])
                result_images = (
                    result
                    if result_images is None
                    else np.vstack([result_images, result])
                )
                # Show action on result image
                cv2.putText(
                    img=result_images,
                    text=f"action: {action}",
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                )

                cv2.imshow("result", result_images)
                key = cv2.waitKey(0)
                if key == 27 & 0xFF:  # ESC key
                    cv2.destroyAllWindows()
                    break

                state = next_state
                hidden_in = hidden_out
                prev_action = common_utils.make_one_hot(
                    torch.as_tensor(action), self.learner.head_cfg.configs.output_size
                )
                prev_reward = torch.as_tensor(reward).to(self.learner.device)
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            if key == 27 & 0xFF:  # ESC key
                break
