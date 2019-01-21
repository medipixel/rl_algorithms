# -*- coding: utf-8 -*-
"""PPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import os
from collections import deque

import git
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

import algorithms.ppo.utils as ppo_utils
from algorithms.ppo.model import Actor, Critic

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "LAMBDA": 0.95,
    "EPSILON": 0.2,
    "W_VALUE": 1.0,
    "W_ENTROPY": 1e-3,
    "ROLLOUT_LENGTH": 256,
    "EPOCH": 8,
    "BATCH_SIZE": 32,
    "MAX_EPISODE_STEPS": 300,
    "EPISODE_NUM": 1500,
}


class Agent(object):
    """PPO Agent.

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        args (dict): arguments including hyperparameters and training settings
        device (torch.device): device selection (cpu / gpu)

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        args (dict): arguments including hyperparameters and training setting
        device (torch.device): device selection (cpu / gpu)
        memory (deque): memory for on-policy training
        transition (list): list for storing a transition
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic

    """

    def __init__(self, env, args, device):
        """Initialization."""
        self.env = env
        self.args = args
        self.device = device
        self.memory = deque()
        self.transition = []
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = float(self.env.action_space.low[0])
        action_high = float(self.env.action_space.high[0])

        # environment setup
        self.env._max_episode_steps = hyper_params["MAX_EPISODE_STEPS"]

        # create models
        self.actor = Actor(state_dim, action_dim, action_low, action_high, device).to(
            device
        )
        self.critic = Critic(state_dim, action_dim, self.device).to(device)

        # create optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # load model parameters
        if self.args.load_from is not None and os.path.exists(self.args.load_from):
            self.load_params(self.args.load_from)

    def select_action(self, state):
        """Select an action from the input space."""
        selected_action, dist = self.actor(state)
        self.transition += [state, dist.log_prob(selected_action).detach()]

        return selected_action

    def step(self, action):
        """Take an action and return the response of the env."""
        action = action.detach().to("cpu").numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.transition += [action, reward, done]
        self.memory.append(self.transition)
        self.transition = []

        return next_state, reward, done

    def update_model(self):
        """Train the model after every N episodes."""
        states, log_probs, actions, rewards, dones = ppo_utils.decompose_memory(
            self.memory
        )

        # calculate returns and gae
        values = self.critic(states)
        returns, advantages = ppo_utils.get_gae(
            rewards, values, dones, hyper_params["GAMMA"], hyper_params["LAMBDA"]
        )

        losses = []
        for state, old_log_prob, action, return_, adv in ppo_utils.ppo_iter(
            hyper_params["EPOCH"],
            hyper_params["BATCH_SIZE"],
            states,
            log_probs,
            actions,
            returns,
            advantages,
        ):
            value = self.critic(state)
            _, dist = self.actor(state)

            # calculate ratios
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            surr_loss = ratio * adv
            clipped_surr_loss = (
                torch.clamp(
                    ratio, 1.0 - hyper_params["EPSILON"], 1.0 + hyper_params["EPSILON"]
                )
                * adv
            )
            actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean()

            # critic_loss
            critic_loss = F.smooth_l1_loss(value, return_)

            # entropy
            entropy = dist.entropy().mean()

            # total_loss
            total_loss = (
                actor_loss
                + hyper_params["W_VALUE"] * critic_loss
                - hyper_params["W_ENTROPY"] * entropy
            )

            # train critic
            self.critic_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()

            losses.append(total_loss.data)

        return sum(losses) / len(losses)

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        if not os.path.exists("./save"):
            os.mkdir("./save")

        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join("./save/ppo_" + sha[:7] + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)
        print("[INFO] saved the model and optimizer to", path)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

        loss = 0
        for i_episode in range(1, hyper_params["EPISODE_NUM"] + 1):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

                if len(self.memory) % hyper_params["ROLLOUT_LENGTH"] == 0:
                    loss = self.update_model()
                    self.memory.clear()

            else:
                print(
                    "[INFO] episode %d\ttotal score: %d\trecent loss: %f"
                    % (i_episode, score, loss)
                )

                if self.args.log:
                    wandb.log({"recent loss": loss, "score": score})

                if i_episode % self.args.save_period == 0:
                    self.save_params(i_episode)

        # termination
        self.env.close()

    def test(self):
        """Test the agent."""
        for i_episode in range(1, hyper_params["EPISODE_NUM"] + 1):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            else:
                print("[INFO] episode %d\ttotal score: %d" % (i_episode, score))

        # termination
        self.env.close()
