"""Config for ApeX-DQN on Pong-No_FrameSkip-v4.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="ApeX",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e5),  # openai baselines: int(1e4)
        batch_size=512,  # openai baselines: 32
        update_starts_from=int(3e4),  # openai baselines: int(1e4)
        multiple_update=1,  # multiple learning updates
        train_freq=1,  # in openai baselines, train_freq = 4
        gradient_clip=10.0,  # dueling: 10.0
        n_step=5,
        w_n_step=1.0,
        w_q_reg=0.0,
        per_alpha=0.6,  # openai baselines: 0.6
        per_beta=0.4,
        per_eps=1e-6,
        loss_type=dict(type="DQNLoss"),
        # Epsilon Greedy
        max_epsilon=1.0,
        min_epsilon=0.1,  # openai baselines: 0.01
        epsilon_decay=5e-7,  # openai baselines: 1e-7 / 1e-1
        # grad_cam
        grad_cam_layer_list=[
            "backbone.cnn.cnn_0.cnn",
            "backbone.cnn.cnn_1.cnn",
            "backbone.cnn.cnn_2.cnn",
        ],
        # ApeX
        num_workers=2,
        local_buffer_max_size=1000,
        worker_update_interval=50,
        logger_interval=1000,
    ),
    learner_cfg=dict(
        type="DQNLearner",
        device="cuda:0",
        backbone=dict(
            type="CNN",
            configs=dict(
                input_sizes=[4, 32, 64],
                output_sizes=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[1, 0, 0],
            ),
        ),
        head=dict(
            type="DuelingMLP",
            configs=dict(
                use_noisy_net=False, hidden_sizes=[512], output_activation=identity
            ),
        ),
        optim_cfg=dict(
            lr_dqn=0.0003,  # dueling: 6.25e-5, openai baselines: 1e-4
            weight_decay=0.0,  # this makes saturation in cnn weights
            adam_eps=1e-8,  # rainbow: 1.5e-4, openai baselines: 1e-8
        ),
    ),
    worker_cfg=dict(type="DQNWorker", device="cpu",),
    logger_cfg=dict(type="DQNLogger",),
    comm_cfg=dict(
        learner_buffer_port=6554,
        learner_worker_port=6555,
        worker_buffer_port=6556,
        learner_logger_port=6557,
        send_batch_port=6558,
        priorities_port=6559,
    ),
)
