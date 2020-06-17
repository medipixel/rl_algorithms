"""Config for DDPGfD on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
import torch

from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="DDPGfDAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e5),
        batch_size=128,
        initial_random_action=int(1e4),
        multiple_update=1,  # multiple learning updates
        gradient_clip_ac=0.5,
        gradient_clip_cr=1.0,
        # fD
        n_step=1,
        pretrain_step=int(5e3),
        lambda1=1.0,  # N-step return weight
        # lambda2 = weight_decay
        lambda3=1.0,  # actor loss contribution of prior weight
        per_alpha=0.3,
        per_beta=1.0,
        per_eps=1e-6,
        per_eps_demo=1.0,
    ),
    learner_cfg=dict(
        type="DDPGfDLearner",
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
        optim_cfg=dict(lr_actor=3e-4, lr_critic=3e-4, weight_decay=1e-4),
    ),
    noise_cfg=dict(ou_noise_theta=0.0, ou_noise_sigma=0.0),
)
