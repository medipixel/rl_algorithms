# -*- coding: utf-8 -*-
"""PPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

from typing import Tuple, Union

import gym
import numpy as np
import torch
import wandb

from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.env.utils import env_generator, make_envs
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict


@AGENTS.register_module
class PPOAgent(Agent):
    """PPO Agent.

    Attributes:
        env (gym.Env): openAI Gym environment
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        actor_optim (Optimizer): optimizer for training actor
        critic_optim (Optimizer): optimizer for training critic
        episode_steps (np.ndarray): step numbers of the current episode
        states (list): memory for experienced states
        actions (list): memory for experienced actions
        rewards (list): memory for experienced rewards
        values (list): memory for experienced values
        masks (list): memory for masks
        log_probs (list): memory for log_probs
        i_episode (int): current episode number
        epsilon (float): value for clipping loss

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
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

        Agent.__init__(
            self,
            env,
            env_info,
            log_cfg,
            is_test,
            load_from,
            is_render,
            render_after,
            is_log,
            save_period,
            episode_num,
            max_episode_steps,
            interim_test_num,
        )

        env_multi = (
            env
            if is_test
            else self.make_parallel_env(max_episode_steps, hyper_params.n_workers)
        )

        self.episode_steps = np.zeros(hyper_params.n_workers, dtype=np.int)
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.masks: list = []
        self.log_probs: list = []
        self.i_episode = 0
        self.next_state = np.zeros((1,))

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg

        if not self.is_test:
            self.env = env_multi

        self.epsilon = hyper_params.max_epsilon

        output_size = (
            self.env_info.action_space.n
            if self.is_discrete
            else self.env_info.action_space.shape[0]
        )

        build_args = dict(
            hyper_params=self.hyper_params,
            log_cfg=self.log_cfg,
            env_name=self.env_info.name,
            state_size=self.env_info.observation_space.shape,
            output_size=output_size,
            is_test=self.is_test,
            load_from=self.load_from,
        )
        self.learner = build_learner(self.learner_cfg, build_args)

    def make_parallel_env(self, max_episode_steps, n_workers):
        if "env_generator" in self.env_info.keys():
            env_gen = self.env_info.env_generator
        else:
            env_gen = env_generator(self.env.spec.id, max_episode_steps)
        env_multi = make_envs(env_gen, n_envs=n_workers)
        return env_multi

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        with torch.no_grad():
            state = numpy2floattensor(state, self.learner.device)
            selected_action, dist = self.learner.actor(state)
            selected_action = selected_action.detach()
            log_prob = dist.log_prob(selected_action)
            value = self.learner.critic(state)

            if self.is_test:
                selected_action = (
                    dist.logits.argmax() if self.is_discrete else dist.mean
                )

            else:
                _selected_action = (
                    selected_action.unsqueeze(1)
                    if self.is_discrete
                    else selected_action
                )
                _log_prob = log_prob.unsqueeze(1) if self.is_discrete else log_prob
                self.states.append(state)
                self.actions.append(_selected_action)
                self.values.append(value)
                self.log_probs.append(_log_prob)

        return selected_action.detach().cpu().numpy()

    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            # if the last state is not a terminal state, store done as false
            done_bool = done.copy()
            done_bool[np.where(self.episode_steps == self.max_episode_steps)] = False

            self.rewards.append(
                numpy2floattensor(reward, self.learner.device).unsqueeze(1)
            )
            self.masks.append(
                numpy2floattensor((1 - done_bool), self.learner.device).unsqueeze(1)
            )

        return next_state, reward, done, info

    def decay_epsilon(self, t: int = 0):
        """Decay epsilon until reaching the minimum value."""
        max_epsilon = self.hyper_params.max_epsilon
        min_epsilon = self.hyper_params.min_epsilon
        epsilon_decay_period = self.hyper_params.epsilon_decay_period

        self.epsilon = self.epsilon - (max_epsilon - min_epsilon) * min(
            1.0, t / (epsilon_decay_period + 1e-7)
        )

    def write_log(
        self,
        log_value: tuple,
    ):
        i_episode, n_step, score, actor_loss, critic_loss, total_loss = log_value
        print(
            "[INFO] episode %d\tepisode steps: %d\ttotal score: %d\n"
            "total loss: %f\tActor loss: %f\tCritic loss: %f\n"
            % (i_episode, n_step, score, total_loss, actor_loss, critic_loss)
        )

        if self.is_log:
            wandb.log(
                {
                    "total loss": total_loss,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "score": score,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.actor, self.critic], log="parameters")

        score = 0
        i_episode_prev = 0
        loss = [0.0, 0.0, 0.0]
        state = self.env.reset()

        while self.i_episode <= self.episode_num:
            for _ in range(self.hyper_params.rollout_len):
                if self.is_render and self.i_episode >= self.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.episode_steps += 1

                state = next_state
                score += reward[0]
                i_episode_prev = self.i_episode
                self.i_episode += done.sum()

                if (self.i_episode // self.save_period) != (
                    i_episode_prev // self.save_period
                ):
                    self.learner.save_params(self.i_episode)

                if done[0]:
                    n_step = self.episode_steps[0]
                    log_value = (
                        self.i_episode,
                        n_step,
                        score,
                        loss[0],
                        loss[1],
                        loss[2],
                    )
                    self.write_log(log_value)
                    score = 0

                self.episode_steps[np.where(done)] = 0
            self.next_state = next_state
            loss = self.learner.update_model(
                (
                    self.states,
                    self.actions,
                    self.rewards,
                    self.values,
                    self.log_probs,
                    self.next_state,
                    self.masks,
                ),
                self.epsilon,
            )
            self.states, self.actions, self.rewards = [], [], []
            self.values, self.masks, self.log_probs = [], [], []
            self.decay_epsilon(self.i_episode)

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
