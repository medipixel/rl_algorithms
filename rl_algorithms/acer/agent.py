import argparse
from typing import Tuple

import gym
import numpy as np
import torch
from torch.distributions import Categorical
import wandb

from rl_algorithms.acer.buffer import ReplayMemory
from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict


@AGENTS.register_module
class ACERAgent(Agent):
    """Discrete Actor Critic with Experience Replay interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        model (nn.Module): policy and value network
        optim (Optimizer): optimizer for model
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        transition (list): recent transition information

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        log_cfg: ConfigDict,
    ):
        Agent.__init__(self, env, env_info, args, log_cfg)

        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.learner_cfg.args = self.args
        self.learner_cfg.env_info = self.env_info
        self.learner_cfg.hyper_params = self.hyper_params
        self.learner_cfg.log_cfg = self.log_cfg

        self.learner = build_learner(self.learner_cfg)
        self.memory = ReplayMemory(self.hyper_params.buffer_size)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action from input space."""
        state = numpy2floattensor(state, self.learner.device)
        with torch.no_grad():
            prob = self.learner.model.pi(state, 0)
        selected_action = Categorical(prob).sample().item()
        return selected_action, prob.squeeze().detach().cpu().numpy()

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the reponse of the env"""
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def write_log(self, log_value: tuple):
        i, score, loss = log_value

        print(
            f"[INFO] episode {i}\t episode step: {self.episode_step} \
            \t total score: {score:.2f}\t loss : {loss:.4f}"
        )

        if self.args.log:
            wandb.log({"loss": loss, "score": score})

    def train(self):
        """Train the agent."""
        if self.args.log:
            self.set_wandb()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            self.episode_step = 0
            while not done:
                seq_data = []
                for _ in range(self.hyper_params.n_rollout):
                    if self.args.render and self.i_episode >= self.args.render_after:
                        self.env.render()

                    action, prob = self.select_action(state)
                    next_state, reward, done, _ = self.step(action)
                    done_mask = 0.0 if done else 1.0
                    self.episode_step += 1
                    transition = (state, action, reward / 100.0, prob, done_mask)
                    seq_data.append(transition)
                    state = next_state
                    score += reward
                    if done:
                        break

                self.memory.add(seq_data)

                if len(self.memory) > self.hyper_params.start_from:
                    experience = self.memory.sample(on_policy=True)
                    self.learner.update_model(experience)
                    n = np.random.poisson(self.hyper_params.replay_ratio)
                    for _ in range(n):
                        experience = self.memory.sample(on_policy=False)
                        loss = self.learner.update_model(experience)
                        loss_episode.append(loss.detach().cpu().numpy())

            loss = np.array(loss_episode).mean()
            log_value = self.i_episode, score, loss
            self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.learner.save_params(self.i_episode)

        self.env.close()
        self.learner.save_params(self.i_episode)
