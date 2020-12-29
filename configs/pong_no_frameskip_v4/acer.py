"""Config for A2C on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="ACERAgent",
    hyper_params=dict(gamma=0.98, c=1, buffer_size=10000),
    learner_cfg=dict(
        type="ACERLearner",
        backbone=dict(
            actor=dict(
                type="CNN",
                configs=dict(
                    input_sizes=[4, 32, 64],
                    output_sizes=[32, 64, 64],
                    kernel_sizes=[8, 4, 3],
                    strides=[4, 2, 1],
                    paddings=[1, 0, 0],
                ),
            ),
            critic=dict(
                type="CNN",
                configs=dict(
                    input_sizes=[4, 32, 64],
                    output_sizes=[32, 64, 64],
                    kernel_sizes=[8, 4, 3],
                    strides=[4, 2, 1],
                    paddings=[1, 0, 0],
                ),
            ),
        ),
        head=dict(
            actor=dict(
                type="SoftmaxHead",
                configs=dict(hidden_sizes=[256, 256], output_activation="identity",),
            ),
            critic=dict(
                type="SoftmaxHead",
                configs=dict(hidden_sizes=[256, 256], output_activation="identity",),
            ),
        ),
        optim_cfg=dict(lr_actor=4e-5, lr_critic=3e-4, weight_decay=0.0),
    ),
)
