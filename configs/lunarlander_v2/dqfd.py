"""Config for DQfD on LunarLander-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="DQfDAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e5),  # openai baselines: int(1e4)
        batch_size=64,  # openai baselines: 32
        update_starts_from=int(1e3),  # openai baselines: int(1e4)
        multiple_update=1,  # multiple learning updates
        train_freq=8,  # in openai baselines, train_freq = 4
        gradient_clip=0.5,  # dueling: 10.0
        n_step=3,
        w_n_step=1.0,
        w_q_reg=1e-7,
        per_alpha=0.6,  # openai baselines: 0.6
        per_beta=0.4,
        per_eps=1e-3,
        # fD
        per_eps_demo=1.0,
        lambda1=1.0,  # N-step return weight
        lambda2=1.0,  # Supervised loss weight
        # lambda3 = weight_decay (l2 regularization weight)
        margin=0.8,
        pretrain_step=int(1e2),
        loss_type=dict(type="C51Loss"),
        # Epsilon Greedy
        max_epsilon=1.0,
        min_epsilon=0.01,  # openai baselines: 0.01
        epsilon_decay=2e-5,  # openai baselines: 1e-7 / 1e-1
    ),
    learner_cfg=dict(
        type="DQfDLearner",
        backbone=dict(),
        head=dict(
            type="C51DuelingMLP",
            configs=dict(
                hidden_sizes=[128, 64],
                use_noisy_net=False,
                v_min=-300,
                v_max=300,
                atom_size=1530,
                output_activation=identity,
            ),
        ),
        optim_cfg=dict(lr_dqn=1e-4, weight_decay=1e-5, adam_eps=1e-8),
    ),
)
