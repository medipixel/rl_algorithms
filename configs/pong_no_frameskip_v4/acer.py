agent = dict(
    type="ACERAgent",
    hyper_params=dict(
        gamma=0.98, c=10, buffer_size=5000, n_rollout=20, replay_ratio=4, start_from=100,
    ),
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
                type="MLP",
                configs=dict(hidden_sizes=[512], output_activation="identity",),
            ),
            critic=dict(
                type="MLP",
                configs=dict(hidden_sizes=[512], output_activation="identity"),
            ),
        ),
        optim_cfg=dict(
            lr=7e-4,
            weight_decay=0.0,
            adam_eps= 1e-8,  # rainbow: 1.5e-4, openai baselines: 1e-8
        ),
    ),
)
