"""Config for A2C on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="A2CAgent",
    params=dict(
        gamma=0.99,
        w_entropy=1e-3,  # multiple learning updates
        gradient_clip_ac=0.1,
        gradient_clip_cr=0.5,
    ),
    network_cfg=dict(hidden_sizes_actor=[256, 256], hidden_sizes_critic=[256, 256]),
    optim_cfg=dict(lr_actor=4e-5, lr_critic=3e-4, weight_decay=0.0),
)
