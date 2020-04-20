from rl_algorithms.utils import Registry, build_from_cfg
from rl_algorithms.utils.config import ConfigDict

AGENTS = Registry("agents")
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
HERS = Registry("hers")


def build_agent(cfg: ConfigDict, build_args: dict = None):
    """Build agent using config and additional arguments."""
    return build_from_cfg(cfg, AGENTS, build_args)


def build_backbone(cfg: ConfigDict, build_args: dict = None):
    """Build agent using config and additional arguments."""
    return build_from_cfg(cfg, BACKBONES, build_args)


def build_head(cfg: ConfigDict, build_args: dict = None):
    """Build agent using config and additional arguments."""
    return build_from_cfg(cfg, HEADS, build_args)


def build_her(cfg: ConfigDict, build_args: dict = None):
    """Build her using config and additional arguments."""
    return build_from_cfg(cfg, HERS, build_args)
