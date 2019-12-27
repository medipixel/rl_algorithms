"""Config for SACfD on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="SACfDAgent",
    params=dict(
        gamma=0.99,
        tau=1e-3,
        buffer_size=int(1e5),
        batch_size=64,
        initial_random_action=int(1e4),
        multiple_learn=2,  # multiple learning updates
        policy_update_freq=2,
        w_entropy=1e-3,
        w_mean_reg=1e-3,
        w_std_reg=1e-3,
        w_pre_activation_reg=0.0,
        auto_entropy_tuning=True,
        # fD
        n_step=3,
        pretrain_step=100,
        lambda1=1.0,  # N-step return weight
        # lambda2 = weight_decay
        lambda3=1.0,  # actor loss contribution of prior weight
        per_alpha=0.6,
        per_beta=0.4,
        per_eps=1e-6,
        per_eps_demo=1.0,
    ),
    network_cfg=dict(
        hidden_sizes_actor=[256, 256],
        hidden_sizes_vf=[256, 256],
        hidden_sizes_qf=[256, 256],
    ),
    optim_cfg=dict(
        lr_actor=3e-4,
        lr_vf=3e-4,
        lr_qf1=3e-4,
        lr_qf2=3e-4,
        lr_entropy=3e-4,
        weight_decay=1e-5,
    ),
)
