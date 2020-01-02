from algorithms.utils import Registry, build_from_cfg
from algorithms.utils.config import ConfigDict

AGENTS = Registry("agents")
HERS = Registry("hers")


def build_agent(cfg: ConfigDict, default_args: dict = None):
    return build_from_cfg(cfg, AGENTS, default_args)


def build_her(cfg: ConfigDict, default_args: dict = None):
    return build_from_cfg(cfg, HERS, default_args)
