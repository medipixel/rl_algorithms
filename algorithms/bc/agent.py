# -*- coding: utf-8 -*-
"""DDPG with Behavior Cloning agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1709.10089.pdf
"""


import os
import pickle

import git
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from algorithms.ddpg.model import Actor, Critic
from algorithms.noise import OUNoise
from algorithms.replay_buffer import ReplayBuffer

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 1024,
    "DEMO_BATCH_SIZE": 128,
    "MAX_EPISODE_STEPS": 300,
    "EPISODE_NUM": 1500,
    "LAMBDA1": 1e-3,
    "LAMBDA2": 1.0,
    "DEMO_PATH": "data/lunarlander_continuous_demo.pkl",
}


class Agent(object):
    """ActorCritic interacting with environment.

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        args (dict): arguments including hyperparameters and training settings
        device (torch.device): device selection (cpu / gpu)

    Attributes:
        env (gym.Env): openAI Gym environment with continuous action space
        actor_local (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic_local (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        args (dict): arguments including hyperparameters and training settings
        noise (OUNoise): random noise for exploration
        memory (ReplayBuffer): replay memory
        device (torch.device): device selection (cpu / gpu)

    """

    def __init__(self, env, args, device):
        """Initialization."""
        self.env = env
        self.args = args
        self.device = device
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = float(self.env.action_space.low[0])
        action_high = float(self.env.action_space.high[0])

        # environment setup
        self.env._max_episode_steps = hyper_params["MAX_EPISODE_STEPS"]

        # create actor
        self.actor_local = Actor(
            state_dim, action_dim, action_low, action_high, self.device
        ).to(device)
        self.actor_target = Actor(
            state_dim, action_dim, action_low, action_high, self.device
        ).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())

        # create critic
        self.critic_local = Critic(state_dim, action_dim, self.device).to(device)
        self.critic_target = Critic(state_dim, action_dim, self.device).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # create optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3)

        # set hyper parameters
        self.lambda1 = hyper_params["LAMBDA1"]
        self.lambda2 = hyper_params["LAMBDA2"] / hyper_params["DEMO_BATCH_SIZE"]

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # noise instance to make randomness of action
        self.noise = OUNoise(action_dim, self.args.seed, theta=0.0, sigma=0.0)

        # replay memory
        self.memory = ReplayBuffer(
            hyper_params["BUFFER_SIZE"],
            hyper_params["BATCH_SIZE"],
            self.args.seed,
            self.device,
        )

        # load demo replay memory
        with open(hyper_params["DEMO_PATH"], "rb") as f:
            demo = pickle.load(f)

        self.demo_memory = ReplayBuffer(
            len(demo),
            hyper_params["DEMO_BATCH_SIZE"],
            self.args.seed,
            self.device,
            demo,
        )

    def select_action(self, state):
        """Select an action from the input space."""
        selected_action = self.actor_local(state)
        selected_action += torch.tensor(self.noise.sample()).float().to(self.device)

        action_low = float(self.env.action_space.low[0])
        action_high = float(self.env.action_space.high[0])
        selected_action = torch.clamp(selected_action, action_low, action_high)

        return selected_action

    def step(self, state, action):
        """Take an action and return the response of the env."""
        action = action.detach().to("cpu").numpy()
        next_state, reward, done, _ = self.env.step(action)
        self.memory.add(state, action, reward, next_state, done)

        return next_state, reward, done

    def update_model(self, experiences, demo):
        """Train the model after each episode."""
        exp_states, exp_actions, exp_rewards, exp_next_states, exp_dones = experiences
        demo_states, demo_actions, demo_rewards, demo_next_states, demo_dones = demo

        states = torch.cat((exp_states, demo_states), dim=0)
        actions = torch.cat((exp_actions, demo_actions), dim=0)
        rewards = torch.cat((exp_rewards, demo_rewards), dim=0)
        next_states = torch.cat((exp_next_states, demo_next_states), dim=0)
        dones = torch.cat((exp_dones, demo_dones), dim=0)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        curr_returns = rewards + (hyper_params["GAMMA"] * next_values * masks)
        curr_returns = curr_returns.to(self.device)

        # crittic loss
        values = self.critic_local(states, actions)
        critic_loss = F.mse_loss(values, curr_returns)

        # train critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # policy loss
        actions = self.actor_local(states)
        policy_loss = -self.critic_local(states, actions).mean()

        # bc loss
        pred_actions = self.actor_local(demo_states)
        qf_mask = torch.gt(
            self.critic_local(demo_states, demo_actions),
            self.critic_local(demo_states, pred_actions),
        ).to(self.device)
        qf_mask = qf_mask.float()
        bc_loss = F.mse_loss(
            torch.mul(pred_actions, qf_mask), torch.mul(demo_actions, qf_mask)
        )

        # train actor: pg loss + BC loss
        actor_loss = self.lambda1 * policy_loss + self.lambda2 * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)

        # for logging
        total_loss = critic_loss + actor_loss

        return total_loss.data

    def soft_update(self, local, target):
        """Soft-update: target = tau*local + (1-tau)*target."""
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(
                hyper_params["TAU"] * l_param.data
                + (1.0 - hyper_params["TAU"]) * t_param.data
            )

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor_local.load_state_dict(params["actor_local_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic_local.load_state_dict(params["critic_local_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        if not os.path.exists("./save"):
            os.mkdir("./save")

        params = {
            "actor_local_state_dict": self.actor_local.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_local_state_dict": self.critic_local.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join("./save/bc_" + sha[:7] + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)
        print("[INFO] saved the model and optimizer to", path)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor_local, self.critic_local], log="parameters")

        for i_episode in range(1, hyper_params["EPISODE_NUM"] + 1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(state, action)

                if len(self.memory) >= hyper_params["BATCH_SIZE"]:
                    experiences = self.memory.sample()
                    demos = self.demo_memory.sample()
                    loss = self.update_model(experiences, demos)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward

            else:
                if len(loss_episode) > 0:
                    avg_loss = np.array(loss_episode).mean()
                    print(
                        "[INFO] episode %d\ttotal score: %d\tloss: %f"
                        % (i_episode, score, avg_loss)
                    )

                    if self.args.log:
                        wandb.log({"score": score, "avg_loss": avg_loss})

                    if i_episode % self.args.save_period == 0:
                        self.save_params(i_episode)

        # termination
        self.env.close()

    def test(self):
        """Test the agent."""
        for i_episode in range(hyper_params["EPISODE_NUM"]):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(state, action)

                state = next_state
                score += reward

            else:
                print("[INFO] episode %d\ttotal score: %d" % (i_episode, score))

        # termination
        self.env.close()
