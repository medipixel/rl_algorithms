from algorithms.utils import Registry, build_from_cfg

AGENTS = Registry("agents")


def build_agent(cfg, default_args=None):
    return build_from_cfg(cfg, AGENTS, default_args)
