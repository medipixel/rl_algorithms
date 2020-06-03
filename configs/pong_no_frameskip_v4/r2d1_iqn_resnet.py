"""Config for R2D1IQN_ResNet on PongNoFrameSkip-v4.
- Author: Kyunghwan Kim, Euijin Jeong
- Contact: kh.kim@medipixel.io, euijin.jeong@medipixel.io
"""
from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="R2D1Agent",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(4e3),  # openai baselines: int(1e4)
        batch_size=32,  # openai baselines: 32
        update_starts_from=int(4e3),  # openai baselines: int(1e4)
        multiple_update=1,  # multiple learning updates
        train_freq=4,  # in openai baselines, train_freq = 4
        gradient_clip=10.0,  # dueling: 10.0
        n_step=3,
        w_n_step=1.0,
        w_q_reg=1e-7,
        per_alpha=0.6,  # openai baselines: 0.6
        per_beta=0.4,
        per_eps=1e-6,
        # R2D1
        sequence_size=20,
        overlap_size=10,
        loss_type=dict(type="R2D1IQNLoss"),
        # Epsilon Greedy
        max_epsilon=1.0,
        min_epsilon=0.01,  # openai baselines: 0.01
        epsilon_decay=1e-6,  # openai baselines: 1e-7 / 1e-1
        # grad_cam
        grad_cam_layer_list=[
            "backbone.cnn.cnn_0.cnn",
            "backbone.cnn.cnn_1.cnn",
            "backbone.cnn.cnn_2.cnn",
        ],
    ),
    backbone=dict(
        type="ResNet",
        configs=dict(
            use_bottleneck=True,
            num_blocks=[1, 1, 1, 1],
            block_output_sizes=[8, 8, 8, 8],
            block_strides=[1, 2, 2, 2],
            first_input_size=1,
            first_output_size=8,
            expansion=1,
            channel_compression=4,  # output channel // channel_compression in last conv layer
        ),
    ),
    head=dict(
        type="IQNMLP",
        configs=dict(
            rnn_hidden_size=512,
            burn_in_step=10,
            hidden_sizes=[512],
            n_tau_samples=64,
            n_tau_prime_samples=64,
            n_quantile_samples=32,
            quantile_embedding_dim=64,
            kappa=1.0,
            use_noisy_net=False,
            output_activation=identity,
        ),
    ),
    optim_cfg=dict(
        lr_dqn=1e-4,  # dueling: 6.25e-5, openai baselines: 1e-4
        weight_decay=0.0,  # this makes saturation in cnn weights
        adam_eps=1e-8,  # rainbow: 1.5e-4, openai baselines: 1e-8
    ),
)
