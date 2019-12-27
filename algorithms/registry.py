from algorithms.utils import Registry, build_from_cfg

AGENTS = Registry("agents")


def build_agent(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, AGENTS, default_args)
