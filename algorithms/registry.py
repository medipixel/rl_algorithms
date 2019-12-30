from algorithms.utils import Registry, build_from_cfg

AGENTS = Registry("agents")
HERS = Registry("hers")


def build_agent(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, AGENTS, default_args)


def build_her(cfg: dict, default_args: dict = None):
    return build_from_cfg(cfg, HERS, default_args)
