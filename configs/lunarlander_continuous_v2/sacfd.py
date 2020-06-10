"""Config for SACfD on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="SACfDAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=1e-3,
        buffer_size=int(1e5),
        batch_size=64,
        initial_random_action=int(5e3),
        multiple_update=2,  # multiple learning updates
        policy_update_freq=2,
        w_entropy=1e-3,
        w_mean_reg=1e-3,
        w_std_reg=1e-3,
        w_pre_activation_reg=0.0,
        auto_entropy_tuning=True,
        # fD
        n_step=3,
        pretrain_step=100,
        lambda1=1.0,  # N-step return weight
        # lambda2 = weight_decay
        lambda3=1.0,  # actor loss contribution of prior weight
        per_alpha=0.6,
        per_beta=0.4,
        per_eps=1e-6,
        per_eps_demo=1.0,
    ),
    learner_cfg=dict(
        type="SACfDLearner",
        backbone=dict(actor=dict(), critic_vf=dict(), critic_qf=dict()),
        head=dict(
            actor=dict(
                type="TanhGaussianDistParams",
                configs=dict(hidden_sizes=[256, 256], output_activation=identity,),
            ),
            critic_vf=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256], output_activation=identity, output_size=1,
                ),
            ),
            critic_qf=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256], output_activation=identity, output_size=1,
                ),
            ),
        ),
        optim_cfg=dict(
            lr_actor=3e-4,
            lr_vf=3e-4,
            lr_qf1=3e-4,
            lr_qf2=3e-4,
            lr_entropy=3e-4,
            weight_decay=1e-5,
        ),
    ),
)
