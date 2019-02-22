<p align="center">

<img src="https://user-images.githubusercontent.com/17582508/52845370-4a930200-314a-11e9-9889-e00007043872.jpg" align="center">

![Build Status](https://travis-ci.org/medipixel/rl_algorithms.svg?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</p>

## Welcome!
This repository contains Reinforcement Learning algorithms which are being used for research activities at Medipixel. The source code will be frequently updated. 
We are warmly welcoming external contributors! :)


## Demo

<p align="center">

|<img src="https://user-images.githubusercontent.com/17582508/52840582-18c76e80-313d-11e9-9752-3d6138f39a15.gif" width="300" height="180"/>|<img src="https://media.giphy.com/media/1mikGEln2lArKMQ6Pt/giphy.gif" width="300" height="180"/>|
|:---:|:---:|
|BC agent on LunarLanderContinuous-v2|SAC agent on Reacher-v2|

</p>

## Contents

1. [REINFORCE](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/reinforce)
2. [Advantage Actor-Critic (A2C)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/a2c)
3. [Deep Deterministic Policy Gradient (DDPG)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/ddpg)
4. [Proximal Policy Optimization Algorithms (PPO)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/ppo)
5. [Twin Delayed Deep Deterministic Policy Gradient Algorithm (TD3)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/td3)
6. [Soft Actor Critic Algorithm (SAC)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/sac/agent.py)
7. [Behaviour Cloning (BC with DDPG)](https://github.com/medipixel/rl_algorithms/tree/master/algorithms/bc)
8. [Prioritized Experience Replay (PER with DDPG)](https://github.com/medipixel/rl_algorithms/tree/master/algorithms/per)
9. [From Demonstrations (DDPGfD, SACfD)](https://github.com/medipixel/rl_algorithms/tree/master/algorithms/fd)
10. [Deep Q-Network (DDQN)](https://github.com/medipixel/rl_algorithms/tree/master/algorithms/dqn)

## Getting started
We have tested each algorithm on some of the following environments.
- [LunarLanderContinuous-v2](https://github.com/medipixel/rl_algorithms/tree/master/examples/lunarlander_continuous_v2)
- [LunarLander_v2](https://github.com/medipixel/rl_algorithms/tree/master/examples/lunarlander_v2)
- [Reacher-v2](https://github.com/medipixel/rl_algorithms/tree/master/examples/reacher-v2)
- [CarRacing-v0](https://github.com/medipixel/rl_algorithms/tree/master/examples/car_racing_v0)


### Prerequisites
In order to run Mujoco environments (e.g. `Reacher-v2`), you need to acquire [Mujoco license](https://www.roboti.us/license.html).

### Installation
First, clone the repository.
```
git clone https://github.com/medipixel/rl_algorithms.git
cd rl_algorithms
```
Secondly, install packages required to execute the code. Just type:
```
make dep
```

#### For developers
You need to type the additional command which configures formatting and linting settings. It automatically runs formatting and linting when you commit the code.

```
make dev
```

After having done `make dev`, you can validate the code by the following commands.
```
make format  # for formatting
make test  # for linting
```

### Usages
You can train or test `algorithm` on `env_name` if `examples/env_name/algorithm.py` exists. (`examples/env_name/algorithm.py` contains hyper-parameters and details of networks.)
```
python run_env_name.py --algo algorithm
``` 

e.g. running soft actor-critic on LunarLanderContinuous-v2.
```
python run_lunarlander_continuous_v2.py --algo sac <other-options>
```

e.g. running a custom agent, **if you have written your own example**: `examples/env_name/ddpg-custom.py`.
```
python run_env_name.py --algo ddpg-custom
```
You will see the agent run with hyper parameter and model settings you configured.

### Argument for run file

In addition, there are various argument settings for running algorithms. If you check the options to run file you should command 
```
python <run-file> -h
```
- `--test`
    - Start test mode (no training).
- `--off-render`
    - Turn off rendering.
- `--log`
    - Turn on logging using [W&B](https://www.wandb.com/).
- `--seed <int>`
    - Set random seed.
- `--save-period <int>`
    - Set saving period of model and optimizer parameters.
- `--max-episode-steps <int>`
    - Set maximum episode step number of the environment. If the number is less than or equal to 0, it uses the default maximum step number of the environment.
- `--episode-num <int>`
    - Set the number of episodes for training.
- `--render-after <int>`
    - Start rendering after the number of episodes.
- `--load-from <save-file-path>`
    - Load the saved models and optimizers at the beginning.

### W&B for logging
We use [W&B](https://www.wandb.com/) for logging of network parameters and others. For more details, read [W&B tutorial](https://docs.wandb.com/docs/started.html).

## References
1. [T. P. Lillicrap et al., "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971, 2015.](https://arxiv.org/pdf/1509.02971.pdf)
2. [J. Schulman et al., "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017.](https://arxiv.org/abs/1707.06347.pdf)
3. [S. Fujimoto et al., "Addressing function approximation error in actor-critic methods." arXiv preprint arXiv:1802.09477, 2018.](https://arxiv.org/pdf/1802.09477.pdf)
4. [T.  Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." arXiv preprint arXiv:1801.01290, 2018.](https://arxiv.org/pdf/1801.01290.pdf)  
5. [T. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications." arXiv preprint arXiv:1812.05905, 2018.](https://arxiv.org/pdf/1812.05905.pdf)
6. [T. Schaul et al., "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
7. [M. Andrychowicz et al., "Hindsight Experience Replay." arXiv preprint arXiv:1707.01495, 2017.](https://arxiv.org/pdf/1707.01495.pdf)
8. [A. Nair et al., "Overcoming Exploration in Reinforcement Learning with Demonstrations." arXiv preprint arXiv:1709.10089, 2017.](https://arxiv.org/pdf/1709.10089.pdf)
9. [M. Vecerik et al., "Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards."arXiv preprint arXiv:1707.08817, 2017](https://arxiv.org/pdf/1707.08817.pdf)
10. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518
(7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
11. [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)