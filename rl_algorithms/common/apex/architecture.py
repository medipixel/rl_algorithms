"""General Ape-X architecture for distributed training.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
- Paper: https://arxiv.org/pdf/1803.00933.pdf
- Reference: https://github.com/haje01/distper
"""
import os

import gym
import ray

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
        args (argparse.Namespace): args from run script
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
        args: ConfigDict,
        env: gym.Env,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        worker_cfg: ConfigDict,
        logger_cfg: ConfigDict,
        comm_cfg: ConfigDict,
        log_cfg: ConfigDict,
    ):
        self.args = args
        self.env = env
        self.env_info = env_info
        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.worker_cfg = worker_cfg
        self.logger_cfg = logger_cfg
        self.comm_cfg = comm_cfg
        self.log_cfg = log_cfg

        self._organize_configs()

        ray.init()

    # pylint: disable=attribute-defined-outside-init
    def _organize_configs(self):
        """Organize configs for initializing components from registry."""
        # organize learner configs
        self.learner_cfg.args = self.args
        self.learner_cfg.env_info = self.env_info
        self.learner_cfg.hyper_params = self.hyper_params
        self.learner_cfg.log_cfg = self.log_cfg
        self.learner_cfg.head.configs.state_size = self.env_info.observation_space.shape
        self.learner_cfg.head.configs.output_size = self.env_info.action_space.n

        # organize worker configs
        self.worker_cfg.env_info = self.env_info
        self.worker_cfg.hyper_params = self.hyper_params
        self.worker_cfg.backbone = self.learner_cfg.backbone
        self.worker_cfg.head = self.learner_cfg.head

        # organize logger configs
        self.logger_cfg.args = self.args
        self.logger_cfg.env_info = self.env_info
        self.logger_cfg.log_cfg = self.log_cfg
        self.logger_cfg.comm_cfg = self.comm_cfg
        self.logger_cfg.backbone = self.learner_cfg.backbone
        self.logger_cfg.head = self.learner_cfg.head

    def _spawn(self):
        """Intialize distributed worker, learner and centralized replay buffer."""
        replay_buffer = ReplayBuffer(
            self.hyper_params.buffer_size, self.hyper_params.batch_size,
        )
        per_buffer = PrioritizedBufferWrapper(
            replay_buffer, alpha=self.hyper_params.per_alpha
        )
        self.global_buffer = ApeXBufferWrapper.remote(
            per_buffer, self.args, self.hyper_params, self.comm_cfg
        )

        learner = build_learner(self.learner_cfg)
        self.learner = ApeXLearnerWrapper.remote(learner, self.comm_cfg)

        state_dict = learner.get_state_dict()
        worker_build_args = dict(args=self.args, state_dict=state_dict)

        self.workers = []
        self.num_workers = self.hyper_params.num_workers
        for rank in range(self.num_workers):
            worker_build_args["rank"] = rank
            worker = build_worker(self.worker_cfg, build_args=worker_build_args)
            apex_worker = ApeXWorkerWrapper.remote(worker, self.args, self.comm_cfg)
            self.workers.append(apex_worker)

        self.logger = build_logger(self.logger_cfg)

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
        if self.args.log:
            worker_logs = [f for f in futures if f is not None]
            self.logger.write_worker_log.remote(
                worker_logs, self.hyper_params.worker_update_interval
            )
        print("Exiting training...")

    def test(self):
        """Load model from checkpoint and run logger for testing."""
        # NOTE: You could also load the Ape-X trained model on the single agent DQN
        self.logger = build_logger(self.logger_cfg)
        self.logger.load_params.remote(self.args.load_from)
        ray.get([self.logger.test.remote(update_step=0, interim_test=False)])
        print("Exiting testing...")
