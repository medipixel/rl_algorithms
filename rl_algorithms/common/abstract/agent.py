# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from abc import ABC, abstractmethod
import argparse
import os
import shutil
import subprocess
from typing import Tuple, Union

import cv2
import gym
from gym.spaces import Discrete
import numpy as np
import torch
import wandb

from rl_algorithms.common.grad_cam import GradCAM
from rl_algorithms.utils.config import ConfigDict


class Agent(ABC):
    """Abstract Agent used for all agents.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        env_name (str) : gym env name for logging
        sha (str): sha code of current git commit
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
        is_discrete (bool): shows whether the action is discrete

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace, log_cfg: ConfigDict):
        """Initialize."""
        self.args = args
        self.env = env
        self.log_cfg = log_cfg

        self.env_name = env.spec.id if env.spec is not None else env.name

        if not self.args.test:
            self.ckpt_path = (
                f"./checkpoint/{self.env_name}/{log_cfg.agent}/{log_cfg.curr_time}/"
            )
            os.makedirs(self.ckpt_path, exist_ok=True)

            # save configuration
            shutil.copy(self.args.cfg_path, os.path.join(self.ckpt_path, "config.py"))

        if isinstance(env.action_space, Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        # for logging
        self.sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])[:-1]
            .decode("ascii")
            .strip()
        )

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def step(
        self, action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    @abstractmethod
    def update_model(self) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def load_params(self, path: str):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

    @abstractmethod
    def save_params(self, params: dict, n_episode: int):
        os.makedirs(self.ckpt_path, exist_ok=True)

        path = os.path.join(self.ckpt_path + self.sha + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

        print("[INFO] Saved the model and optimizer to", path)

    @abstractmethod
    def write_log(self, log_value: tuple):  # type: ignore
        pass

    @abstractmethod
    def train(self):
        pass

    def set_wandb(self):
        wandb.init(
            project=self.env_name,
            name=f"{self.log_cfg.agent}/{self.log_cfg.curr_time}",
        )
        wandb.config.update(vars(self.args))
        shutil.copy(self.args.cfg_path, os.path.join(wandb.run.dir, "config.py"))

    def interim_test(self):
        self.args.test = True

        print()
        print("===========")
        print("Start Test!")
        print("===========")

        self._test(interim_test=True)

        print("===========")
        print("Test done!")
        print("===========")
        print()

        self.args.test = False

    def test(self):
        """Test the agent."""
        # logger
        if self.args.log:
            self.set_wandb()

        self._test()

        # termination
        self.env.close()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num

        for i_episode in range(test_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.args.render:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )

            if self.args.log:
                wandb.log({"test score": score})

    def test_with_gradcam(self):
        """Test agent with Grad-CAM."""
        gcam = GradCAM(model=self.dqn.eval())

        for i_episode in range(self.args.episode_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            key = 0
            while not done:
                state = self._preprocess_state(state)
                action = self.dqn(state).argmax()
                action = action.detach().cpu().numpy()
                next_state, reward, done, _ = self.step(action)

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
                location = (50, 50)
                font = cv2.FONT_HERSHEY_PLAIN  # hand-writing style font
                fontScale = 1
                cv2.putText(
                    result_images,
                    f"action: {action}",
                    location,
                    font,
                    fontScale,
                    (0, 0, 255),
                    2,
                )

                cv2.imshow("result", result_images)
                key = cv2.waitKey(0)
                if key == 27 & 0xFF:
                    cv2.destroyAllWindows()
                    break

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            if key == 27 & 0xFF:
                break
