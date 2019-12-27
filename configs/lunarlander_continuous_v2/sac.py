"""Config for SAC on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="SACAgent",
    params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e6),
        batch_size=128,
        initial_random_action=int(1e4),
        multiple_learn=1,  # multiple learning updates
        policy_update_freq=2,
        w_entropy=1e-3,
        w_mean_reg=0.0,
        w_std_reg=0.0,
        w_pre_activation_reg=0.0,
        auto_entropy_tuning=True,
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
        weight_decay=1e-6,
    ),
)
