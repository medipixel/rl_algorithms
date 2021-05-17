"""Config for DQN(IQN) on Pong-No_FrameSkip-v4.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="DistillationDQN",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e4),
        batch_size=256,
        update_starts_from=int(1e4),  # openai baselines: int(1e4)
        multiple_update=1,  # multiple learning updates
        train_freq=4,  # in openai baselines, train_freq = 4
        gradient_clip=10.0,  # dueling: 10.0
        n_step=3,
        w_n_step=1.0,
        w_q_reg=0.0,
        per_alpha=0.6,  # openai baselines: 0.6
        per_beta=0.4,
        per_eps=1e-6,
        # Epsilon Greedy
        max_epsilon=0.0,
        min_epsilon=0.0,  # openai baselines: 0.01
        epsilon_decay=1e-6,  # openai baselines: 1e-7 / 1e-1
        # Grad_cam
        grad_cam_layer_list=[
            "backbone.cnn.cnn_0.cnn",
            "backbone.cnn.cnn_1.cnn",
            "backbone.cnn.cnn_2.cnn",
        ],
        # Distillation
        dataset_path=[
            "data/distillation_buffer/offline_paper/lad2_expertq",
            "data/distillation_buffer/offline_paper/lad3_expertq",
            "data/distillation_buffer/offline_paper/lcx1_expertq",
            "data/distillation_buffer/offline_paper/rca2_expertq",
            "data/distillation_buffer/offline_paper/rca3_expertq",
        ],
        save_dir="data/",
        epochs=200,  # epoch of student training
        n_frame_from_last=int(5e4),  # number of frames to save from the end of training
    ),
    learner_cfg=dict(
        type="DQNLearner",
        loss_type=dict(type="IQNLoss"),
        backbone=dict(
            type="ResNet",
            configs=dict(
                use_bottleneck=False,
                num_blocks=[1, 1, 1, 1],
                block_output_sizes=[32, 32, 64, 64],
                block_strides=[1, 2, 2, 2],
                first_input_size=4,
                first_output_size=32,
                expansion=1,
                channel_compression=4,  # compression ratio
            ),
        ),
        head=dict(
            type="C51DuelingMLP",
            configs=dict(
                hidden_sizes=[512],
                use_noisy_net=False,
                v_min=-2.0,
                v_max=0.0,
                atom_size=51,
                output_activation="identity",
            ),
        ),
        optim_cfg=dict(
            lr_dqn=1e-4,  # dueling: 6.25e-5, openai baselines: 1e-4
            weight_decay=0.0,  # this makes saturation in cnn weights
            adam_eps=1e-8,  # rainbow: 1.5e-4, openai baselines: 1e-8
        ),
    ),
)
