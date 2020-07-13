"""Config for DQN(IQN) on Pong-No_FrameSkip-v4.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="DistillationDQN",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
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
        loss_type=dict(type="IQNLoss"),
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
        epochs=20,  # epoch of student training
        buffer_size=int(50000),  # distillation buffer size
        batch_size=32,  # distillation batch size
    ),
    learner_cfg=dict(
        type="DQNLearner",
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
            type="IQNMLP",
            configs=dict(
                hidden_sizes=[512],
                n_tau_samples=64,
                n_tau_prime_samples=64,
                n_quantile_samples=32,
                quantile_embedding_dim=64,
                kappa=1.0,
                output_activation=identity,
                # NoisyNet
                use_noisy_net=True,
                std_init=0.5,
            ),
        ),
        optim_cfg=dict(
            lr_dqn=1e-4,  # dueling: 6.25e-5, openai baselines: 1e-4
            weight_decay=0.0,  # this makes saturation in cnn weights
            adam_eps=1e-8,  # rainbow: 1.5e-4, openai baselines: 1e-8
        ),
    ),
)
