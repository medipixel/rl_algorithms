# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from abc import ABC, abstractmethod
import os
import shutil
from typing import Tuple, Union

import cv2
import gym
import numpy as np
import torch
import wandb

from rl_algorithms.common.grad_cam import GradCAM
from rl_algorithms.common.saliency_map import make_saliency_dir, save_saliency_maps
from rl_algorithms.utils.config import ConfigDict


class Agent(ABC):
    """
    Abstract Agent used for all agents.

    Attributes:
        env (gym.Env): openAI Gym environment
        log_cfg (ConfigDict): configuration for saving log
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        log_cfg: ConfigDict,
        is_test: bool,
        load_from: str,
        is_render: bool,
        render_after: int,
        is_log: bool,
        save_period: int,
        episode_num: int,
        max_episode_steps: int,
        interim_test_num: int,
    ):
        """Initialize."""
        self.env = env
        self.env_info = env_info
        self.log_cfg = log_cfg

        self.is_test = is_test
        self.load_from = load_from
        self.is_render = is_render
        self.render_after = render_after
        self.is_log = is_log
        self.save_period = save_period
        self.episode_num = episode_num
        self.max_episode_steps = max_episode_steps
        self.interim_test_num = interim_test_num

        self.total_step = 0
        self.learner = None

        self.is_discrete = isinstance(self.env_info.action_space, gym.spaces.Discrete)

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def step(
        self, action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    @abstractmethod
    def write_log(self, log_value: tuple):  # type: ignore
        pass

    @abstractmethod
    def train(self):
        pass

    def set_wandb(self):
        """Set configuration for wandb logging."""
        wandb.init(
            project=self.env_info.name,
            name=f"{self.log_cfg.agent}/{self.log_cfg.curr_time}",
        )
        additional_log = dict(
            save_period=self.save_period,
            episode_num=self.episode_num,
            max_episode_steps=self.max_episode_steps,
        )
        wandb.config.update(additional_log)
        wandb.config.update(self.hyper_params)
        shutil.copy(self.log_cfg.cfg_path, os.path.join(wandb.run.dir, "config.yaml"))

    def interim_test(self):
        """Test in the middle of training."""
        self.is_test = True

        print()
        print("===========")
        print("Start Test!")
        print("===========")

        self._test(interim_test=True)

        print("===========")
        print("Test done!")
        print("===========")
        print()

        self.is_test = False

    def test(self):
        """Test the agent."""
        # logger
        if self.is_log:
            self.set_wandb()

        self._test()

        # termination
        self.env.close()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.interim_test_num
        else:
            test_num = self.episode_num

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

    def test_with_gradcam(self):
        """Test agent with Grad-CAM."""
        policy = self.learner.get_policy()
        gcam = GradCAM(model=policy.eval())

        for i_episode in range(self.episode_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            key = 0
            print("\nPress Any Key to move to next step... (quit: ESC key)")
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)

                state = self._preprocess_state(state)
                _ = gcam.forward(state)
                ids = torch.LongTensor([[int(action)]]).cuda()
                gcam.backward(ids=ids)

                state = state[-1].detach().cpu().numpy().astype(np.uint8)
                state = np.transpose(state)
                state = cv2.cvtColor(state, cv2.COLOR_GRAY2BGR)
                state = cv2.resize(state, (150, 150), interpolation=cv2.INTER_LINEAR)

                # Get Grad-CAM image
                result_images = None
                for target_layer in self.hyper_params.grad_cam_layer_list:
                    regions = gcam.generate(target_layer)
                    regions = regions.detach().cpu().numpy()
                    regions = np.squeeze(regions) * 255
                    regions = np.transpose(regions)
                    regions = cv2.applyColorMap(
                        regions.astype(np.uint8), cv2.COLORMAP_JET
                    )
                    regions = cv2.resize(
                        regions, (150, 150), interpolation=cv2.INTER_LINEAR
                    )
                    overlay = cv2.addWeighted(state, 1.0, regions, 0.5, 0)
                    result = np.hstack([state, regions, overlay])
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
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            if key == 27 & 0xFF:  # ESC key
                break

    def test_with_saliency_map(self):
        """Test agent with saliency map."""
        saliency_map_dir = make_saliency_dir(self.args.load_from.split("/")[-2])
        print(f"Save saliency map in directory : {saliency_map_dir}")
        print("Saving saliency maps...")
        i = 0
        for i_episode in range(self.args.episode_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            key = 0
            print("\nPress Any Key to move to next step... (quit: ESC key)")
            while not done:
                action = self.select_action(state)
                for param in self.learner.dqn.parameters():
                    param.requires_grad = False
                saliency_map = save_saliency_maps(
                    i,
                    state,
                    action,
                    self.learner.dqn,
                    self.learner.device,
                    saliency_map_dir,
                )
                i += 1
                next_state, reward, done, _ = self.step(action)

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
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            if key == 27 & 0xFF:  # ESC key
                break
