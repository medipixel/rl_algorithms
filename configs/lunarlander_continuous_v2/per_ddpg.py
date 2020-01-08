"""Config for DDPG with PER on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="PERDDPGAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e6),
        batch_size=128,
        initial_random_action=int(1e4),
        multiple_update=1,  # multiple learning updates
        gradient_clip_ac=0.5,
        gradient_clip_cr=1.0,
        # PER
        per_alpha=0.6,
        per_beta=0.4,
        per_eps=1e-6,
    ),
    network_cfg=dict(hidden_sizes_actor=[256, 256], hidden_sizes_critic=[256, 256]),
    optim_cfg=dict(lr_actor=3e-4, lr_critic=3e-4, weight_decay=5e-6),
    noise_cfg=dict(ou_noise_theta=0.0, ou_noise_sigma=0.0),
)
