"""Config for A2C on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="ACERAgent",
    hyper_params=dict(gamma=0.98, c=1, buffer_size=5000,),
    learner_cfg=dict(
        type="ACERLearner",
        backbone=dict(actor=None, critic=None),
        head=dict(
            actor=dict(
                type="SoftmaxHead",
                configs=dict(hidden_sizes=[256, 256], output_activation="identity",),
            ),
            critic=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256],
                    output_activation="identity",
                    output_size=4,
                ),
            ),
        ),
        optim_cfg=dict(lr_actor=0.0002, lr_critic=0.0002, weight_decay=0.0),
    ),
)
