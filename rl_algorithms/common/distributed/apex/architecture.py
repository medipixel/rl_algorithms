import gym
import ray

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper
from rl_algorithms.common.distributed.abstract.architecture import Architecture
from rl_algorithms.common.distributed.apex.buffer import ApeXBufferWrapper
from rl_algorithms.common.distributed.apex.learner import ApeXLearnerWrapper
from rl_algorithms.common.distributed.apex.worker import ApeXWorkerWrapper
from rl_algorithms.registry import AGENTS, build_learner, build_logger, build_worker
from rl_algorithms.utils.config import ConfigDict


@AGENTS.register_module
class ApeX(Architecture):
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
        self._spawn()

    # pylint: disable=attribute-defined-outside-init
    def _organize_configs(self):
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
        # create buffer
        replay_buffer = ReplayBuffer(
            self.hyper_params.buffer_size, self.hyper_params.batch_size,
        )
        per_buffer = PrioritizedBufferWrapper(
            replay_buffer, alpha=self.hyper_params.per_alpha
        )
        self.global_buffer = ApeXBufferWrapper.remote(
            per_buffer, self.args, self.hyper_params, self.comm_cfg
        )

        # create learner
        learner = build_learner(self.learner_cfg)
        self.learner = ApeXLearnerWrapper.remote(learner, self.comm_cfg)

        # create workers
        state_dict = learner.get_state_dict()
        worker_build_args = dict(args=self.args, state_dict=state_dict)

        self.workers = []
        self.num_workers = 4
        for rank in range(self.num_workers):
            worker_build_args["rank"] = rank
            worker = build_worker(self.worker_cfg, build_args=worker_build_args)
            apex_worker = ApeXWorkerWrapper.remote(worker, self.comm_cfg)
            self.workers.append(apex_worker)

        self.logger = build_logger(self.logger_cfg)

        # put together all processes
        self.processes = self.workers + [self.learner, self.global_buffer, self.logger]

    def train(self):
        print("Running main training loop...")
        run_procs = [proc.run.remote() for proc in self.processes]
        finished_proc = ray.wait(run_procs)

        if not finished_proc:
            # Only learner has exit criterion in while loop
            for proc in run_procs:
                ray.cancel(proc, force=True)

        del run_procs
        print("Exiting training...")
