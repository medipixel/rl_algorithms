"""Config for DDPG on Reacher-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
import torch

from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="DDPGAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=1e-3,
        buffer_size=int(1e5),
        batch_size=128,
        initial_random_action=int(1e4),
        multiple_update=1,  # multiple learning updates
        gradient_clip_ac=0.5,
        gradient_clip_cr=1.0,
    ),
    learner_cfg=dict(
        type="DDPGLearner",
        backbone=dict(actor=dict(), critic=dict(),),
        head=dict(
            actor=dict(
                type="MLP",
                configs=dict(hidden_sizes=[256, 256], output_activation=torch.tanh,),
            ),
            critic=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256], output_size=1, output_activation=identity,
                ),
            ),
        ),
        optim_cfg=dict(lr_actor=1e-3, lr_critic=1e-3, weight_decay=1e-6),
        noise_cfg=dict(ou_noise_theta=0.0, ou_noise_sigma=0.0),
    ),
    noise_cfg=dict(ou_noise_theta=0.0, ou_noise_sigma=0.0),
)
