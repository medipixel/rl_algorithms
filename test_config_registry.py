from algorithms import (
    A2CAgent,
    BCDDPGAgent,
    BCSACAgent,
    DDPGAgent,
    DDPGfDAgent,
    DQfDAgent,
    DQNAgent,
    PERDDPGAgent,
    PPOAgent,
    SACAgent,
    SACfDAgent,
    TD3Agent,
    build_agent,
)
from algorithms.utils import Config


def test_config_registry():
    cfg_path = "./configs/lunarlander_continuous_v2/ddpg.py"
    cfg = Config.fromfile(cfg_path)
    agent = build_agent(cfg.agent)
    assert isinstance(agent, DDPGAgent)


if __name__ == "__main__":
    test_config_registry()
