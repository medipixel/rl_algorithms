"""Config for DDPG with Behavior Cloning on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="BCDDPGAgent",
    params=dict(
        gamma=0.99,
        tau=1e-3,
        buffer_size=int(1e5),
        batch_size=512,
        initial_random_action=int(1e4),
        multiple_learn=1,  # multiple learning updates
        gradient_clip_ac=0.5,
        gradient_clip_cr=0.5,
        # BC
        demo_batch_size=64,
        lambda1=1e-3,
        lambda2=1.0,
        # HER
        use_her=False,
        her=dict(type="LunarLanderContinuousHER",),
        success_score=250.0,
        desired_states_from_demo=True,
    ),
    network_cfg=dict(hidden_sizes_actor=[256, 256], hidden_sizes_critic=[256, 256]),
    optim_cfg=dict(lr_actor=1e-4, lr_critic=1e-3, weight_decay=1e-4),
    noise_cfg=dict(ou_noise_theta=0.0, ou_noise_sigma=0.0),
)
