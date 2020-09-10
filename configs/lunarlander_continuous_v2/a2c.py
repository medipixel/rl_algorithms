"""Config for A2C on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
import torch

from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="A2CAgent",
    hyper_params=dict(
        gamma=0.99,
        w_entropy=1e-3,  # multiple learning updates
        gradient_clip_ac=0.1,
        gradient_clip_cr=0.5,
    ),
    learner_cfg=dict(
        type="A2CLearner",
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        backbone=dict(actor=dict(), critic=dict(),),
        head=dict(
            actor=dict(
                type="GaussianDist",
                configs=dict(hidden_sizes=[256, 256], output_activation=identity,),
            ),
            critic=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256], output_activation=identity, output_size=1,
                ),
            ),
        ),
        optim_cfg=dict(lr_actor=4e-5, lr_critic=3e-4, weight_decay=0.0),
    ),
)
