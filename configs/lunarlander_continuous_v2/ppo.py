"""Config for PPO on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
import torch

agent = dict(
    type="PPOAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=0.95,
        batch_size=32,
        max_epsilon=0.2,
        min_epsilon=0.2,
        epsilon_decay_period=1500,
        w_value=1.0,
        w_entropy=1e-3,
        gradient_clip_ac=0.1,
        gradient_clip_cr=0.5,
        epoch=16,
        rollout_len=256,
        n_workers=12,
        use_clipped_value_loss=True,
        standardize_advantage=True,
    ),
    learner_cfg=dict(
        type="PPOLearner",
        backbone=dict(actor=dict(), critic=dict(),),
        head=dict(
            actor=dict(
                type="GaussianDist",
                configs=dict(hidden_sizes=[256, 256], output_activation=torch.tanh,),
            ),
            critic=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256],
                    output_size=1,
                    output_activation=torch.tanh,
                ),
            ),
        ),
        optim_cfg=dict(lr_actor=3e-4, lr_critic=1e-3, weight_decay=0.0),
    ),
)
