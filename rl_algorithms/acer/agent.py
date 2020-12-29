import numpy as np
from torch.distributions import Categorical
import wandb

from rl_algorithms.acer.buffer import ReplayMemory
from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS, build_learner


@AGENTS.register_module
class ACERAgent(Agent):
    def __init__(self, env, env_info, args, hyper_params, learner_cfg, log_cfg):
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

    def select_action(self, state):
        state = numpy2floattensor(state, self.learner.device)
        prob = self.learner.actor(state, 0)
        selected_action = Categorical(prob).sample().item()

        return selected_action, prob.squeeze().detach().cpu().numpy()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def write_log(self, log_value):
        i, score, actor_loss, critic_loss = log_value

        print(
            f"[INFO] episode {i}\t episode step: {self.episode_step}\t total score: {score:.2f}\t actor loss : {actor_loss:.4f}\t critic loss : {critic_loss:.4f}"
        )

        if self.args.log:
            wandb.log(
                {"actor loss": actor_loss, "critic_loss": critic_loss, "score": score,}
            )

    def train(self):
        if self.args.log:
            self.set_wandb()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            actor_loss_episode = list()
            critic_loss_episode = list()

            self.episode_step = 0
            while not done:
                seq_data = []
                for _ in range(10):
                    if self.args.render and self.i_episode >= self.args.render_after:
                        self.env.render()

                    action, prob = self.select_action(state)
                    next_state, reward, done, _ = self.step(action)
                    done_mask = 0.0 if done else 1.0
                    self.episode_step += 1
                    transition = (state, action, reward, prob, done_mask)
                    seq_data.append(transition)
                    state = next_state
                    score += reward

                self.memory.push(seq_data)

                if len(self.memory) > 500:
                    experience = self.memory.sample(16, on_policy=True)
                    self.learner.update_model(experience)
                    experience = self.memory.sample(16)
                    actor_loss, critic_loss = self.learner.update_model(experience)
                    actor_loss_episode.append(actor_loss.detach().cpu().numpy())
                    critic_loss_episode.append(critic_loss.detach().cpu().numpy())

            actor_loss = np.array(actor_loss_episode).mean()
            critic_loss = np.array(critic_loss_episode).mean()
            log_value = self.i_episode, score, actor_loss, critic_loss
            self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.learner.save_params(self.i_episode)

        self.env.close()
        self.learner.save_params(self.i_episode)
