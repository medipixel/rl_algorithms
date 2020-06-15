"""Config for TD3 on Reacher-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
import torch

from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="TD3Agent",
    hyper_params=dict(
        gamma=0.95,
        tau=5e-3,
        buffer_size=int(1e6),
        batch_size=100,
        initial_random_action=int(1e4),
        policy_update_freq=2,
    ),
    learner_cfg=dict(
        type="TD3Learner",
        backbone=dict(actor=dict(), critic=dict()),
        head=dict(
            actor=dict(
                type="MLP",
                configs=dict(hidden_sizes=[400, 300], output_activation=torch.tanh,),
            ),
            critic=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[400, 300], output_size=1, output_activation=identity,
                ),
            ),
        ),
        optim_cfg=dict(lr_actor=1e-3, lr_critic=1e-3, weight_decay=0.0),
    ),
    noise_cfg=dict(
        exploration_noise=0.1, target_policy_noise=0.2, target_policy_noise_clip=0.5
    ),
)
