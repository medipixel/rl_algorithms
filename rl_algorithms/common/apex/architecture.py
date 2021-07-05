"""General Ape-X architecture for distributed training.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
- Paper: https://arxiv.org/pdf/1803.00933.pdf
- Reference: https://github.com/haje01/distper
"""
import os

import gym
import ray
import torch

from rl_algorithms.common.abstract.architecture import Architecture
from rl_algorithms.common.apex.learner import ApeXLearnerWrapper
from rl_algorithms.common.apex.worker import ApeXWorkerWrapper
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.wrapper import (
    ApeXBufferWrapper,
    PrioritizedBufferWrapper,
)
from rl_algorithms.registry import AGENTS, build_learner, build_logger, build_worker
from rl_algorithms.utils.config import ConfigDict

# NOTE: Setting LRU_CACHE_CAPACITY to a low number fixes CPU memory leak.
# For PyTorch <= 1.5, it is necessary to set cache capacity manually.
# See https://github.com/pytorch/pytorch/issues/27971
os.environ["LRU_CACHE_CAPACITY"] = "10"


@AGENTS.register_module
class ApeX(Architecture):
    """General Ape-X architecture for distributed training.

    Attributes:
        rank (int): rank (ID) of worker
        env_info (ConfigDict): information about environment
        hyper_params (ConfigDict): algorithm hyperparameters
        learner_cfg (ConfigDict): configs for learner class
        worker_cfg (ConfigDict): configs for worker class
        logger_cfg (ConfigDict): configs for logger class
        comm_cfg (ConfigDict): configs for inter-process communication
        log_cfg (ConfigDict): configs for logging, passed on to logger_cfg
        learner (Learner): distributed learner class
        workers (list): List of distributed worker class
        global buffer (ReplayBuffer): centralized buffer wrapped with PER and ApeX
        logger (DistributedLogger): logger class
        processes (list): List of all processes

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        worker_cfg: ConfigDict,
        logger_cfg: ConfigDict,
        comm_cfg: ConfigDict,
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
        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.worker_cfg = worker_cfg
        self.logger_cfg = logger_cfg
        self.comm_cfg = comm_cfg
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

        assert (
            torch.cuda.is_available()
        ), "Training with CPU do not guarantee performance. Please run with GPU device."

        ray.init()

    # pylint: disable=attribute-defined-outside-init
    def _spawn(self):
        """Intialize distributed worker, learner and centralized replay buffer."""
        replay_buffer = ReplayBuffer(
            self.hyper_params.buffer_size,
            self.hyper_params.batch_size,
        )
        per_buffer = PrioritizedBufferWrapper(
            replay_buffer, alpha=self.hyper_params.per_alpha
        )
        self.global_buffer = ApeXBufferWrapper.remote(
            per_buffer, self.hyper_params, self.comm_cfg
        )

        # Build learner
        learner_build_args = dict(
            hyper_params=self.hyper_params,
            log_cfg=self.log_cfg,
            env_name=self.env_info.name,
            state_size=self.env_info.observation_space.shape,
            output_size=self.env_info.action_space.n,
            is_test=self.is_test,
            load_from=self.load_from,
        )
        learner = build_learner(self.learner_cfg, learner_build_args)
        self.learner = ApeXLearnerWrapper.remote(learner, self.comm_cfg)

        # Build workers
        state_dict = learner.get_state_dict()
        worker_build_args = dict(
            hyper_params=self.hyper_params,
            backbone=self.learner_cfg.backbone,
            head=self.learner_cfg.head,
            loss_type=self.learner_cfg.loss_type,
            state_dict=state_dict,
            env_name=self.env_info.name,
            state_size=self.env_info.observation_space.shape,
            output_size=self.env_info.action_space.n,
            is_atari=self.env_info.is_atari,
            max_episode_steps=self.max_episode_steps,
        )
        self.workers = []
        self.num_workers = self.hyper_params.num_workers
        for rank in range(self.num_workers):
            worker_build_args["rank"] = rank
            worker = build_worker(self.worker_cfg, build_args=worker_build_args)
            apex_worker = ApeXWorkerWrapper.remote(worker, self.comm_cfg)
            self.workers.append(apex_worker)

        # Build logger
        logger_build_args = dict(
            log_cfg=self.log_cfg,
            comm_cfg=self.comm_cfg,
            backbone=self.learner_cfg.backbone,
            head=self.learner_cfg.head,
            env_name=self.env_info.name,
            is_atari=self.env_info.is_atari,
            state_size=self.env_info.observation_space.shape,
            output_size=self.env_info.action_space.n,
            max_update_step=self.hyper_params.max_update_step,
            episode_num=self.episode_num,
            max_episode_steps=self.max_episode_steps,
            is_log=self.is_log,
            is_render=self.is_render,
            interim_test_num=self.interim_test_num,
        )

        self.logger = build_logger(self.logger_cfg, logger_build_args)

        self.processes = self.workers + [self.learner, self.global_buffer, self.logger]

    def train(self):
        """Spawn processes and run training loop."""
        print("Spawning and initializing communication...")
        # Spawn processes
        self._spawn()

        # Initialize communication
        for proc in self.processes:
            proc.init_communication.remote()

        # Run main training loop
        print("Running main training loop...")
        run_procs = [proc.run.remote() for proc in self.processes]
        futures = ray.get(run_procs)

        # Retreive workers' data and write to wandb
        # NOTE: Logger logs the mean scores of each episode per update step
        if self.is_log:
            worker_logs = [f for f in futures if f is not None]
            self.logger.write_worker_log.remote(
                worker_logs, self.hyper_params.worker_update_interval
            )
        print("Exiting training...")

    def test(self):
        """Load model from checkpoint and run logger for testing."""
        # NOTE: You could also load the Ape-X trained model on the single agent DQN
        self.logger = build_logger(self.logger_cfg)
        self.logger.load_params.remote(self.load_from)
        ray.get([self.logger.test.remote(update_step=0, interim_test=False)])
        print("Exiting testing...")
