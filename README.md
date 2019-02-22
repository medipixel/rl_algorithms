![medipixel_ci](https://user-images.githubusercontent.com/17582508/52845370-4a930200-314a-11e9-9889-e00007043872.jpg)  

![Build Status](https://travis-ci.org/medipixel/rl_algorithms.svg?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

RL key algorithms with [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/) and [Reacher-v2](https://gym.openai.com/envs/Reacher-v2/).

|<img src="https://user-images.githubusercontent.com/17582508/52840582-18c76e80-313d-11e9-9752-3d6138f39a15.gif" width="400"/>|<img src="https://media.giphy.com/media/1mikGEln2lArKMQ6Pt/giphy.gif" width="400"/>|
|:---:|:---:|
|BC agent in LunarLanderContinuous-v2|SAC agent in Reacher-v2|

## Contents

1. [REINFORCE with baseline](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/reinforce)
2. [Advantage Actor-Critic (A2C)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/a2c)
3. [Deep Deterministic Policy Gradient (DDPG)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/ddpg)
4. [Proximal Policy Optimization Algorithms (PPO)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/ppo)
5. [Twin Delayed Deep Deterministic Policy Gradient Algorithm (TD3)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/td3)
6. [Soft Actor Critic Algorithm (SAC)](https://github.com/medipixel/rl_algorithms/blob/master/algorithms/sac/agent.py)
7. [Behaviour Cloning (BC with DDPG)](https://github.com/medipixel/rl_algorithms/tree/master/algorithms/bc)
8. [Prioritized Experience Replay (PER with DDPG)](https://github.com/medipixel/rl_algorithms/tree/master/algorithms/per)
9. [From Demonstrations (DDPGfD, SACfD)](https://github.com/medipixel/rl_algorithms/tree/master/algorithms/fd)

## Getting started
RL algorithms can be run locally with gym environments. Currently our implemented run codes are
- [LunarLanderContinuous-v2](https://github.com/medipixel/rl_algorithms/tree/feature/readme/examples/lunarlander_continuous_v2)
- [Reacher-v2](https://github.com/medipixel/rl_algorithms/tree/feature/readme/examples/reacher-v2)

### Prerequisites
Mujoco environments(e.g. `Reacher-v2`) require a [Mujoco](https://www.roboti.us/license.html) license.

### Installation
First, clone this repository  
```
git clone https://github.com/medipixel/rl_algorithms.git
cd rl_algorithms
```
And, install packages required to run the code. Just command,
```
make dep
```

### Usage example
If you want to train model with `DDPG` in `LunarLanderContinuous-v2` environment,
```
python run_lunarlander_continuous_v2.py --algo ddpg
``` 
If you change **hyper parameters** any algorithms, check the [examples/<env-name> directory](https://github.com/medipixel/rl_algorithms/tree/master/examples). If `examples/<env-name>/<algorithm-name>` path exists, you can run
```
python <run-file> --algo <algorithm-name>
```
You can make new example python file(e.g. `ddpg-custom`), and you can run
```
python <run-file> --algo ddpg-custom
```
You can see to run the algorithm with customized hyper parameters for that you want.  

### Argument for run file

In addition, there are various argument settings for running algorithms. If you check the options to run file you should command 
```
python <run-file> -h
```
- `--test`
    - Set test mode (no training).
- `--off-render`
    - Turn off rendering.
- `--log`
    - Turn on logging using [WanDB](https://www.wandb.com/).
- `--seed <int>`
    - Set random seed.
- `--save-period <int>`
    - Set period of save model.
- `--max-episode-steps <int>`
    - Set max episode steps of the environment. If input is number under 1, set default max steps of the environment.
- `--episode-num <int>`
    - Set number of episode for training.
- `--render-after <int>`
    - Start rendering after the input number of steps.
- `--load-from <save-file-path>`
    - Load the saved model and optimizer at the beginning. When testing a specific saved model that you want, use this argument.
- `--demo-path <demo-file-path(.pkl)>`
    - Set demonstration path for algorithm using demonstration such as Behavior cloning.

### Development setup
Describe how to install all development dependencies and how to run an automated check of python code.
```
make dev
```

### Wandb log
We use [WanDB](https://www.wandb.com/) to log network parameters and other variables. You should read a [wandb manual](https://docs.wandb.com/docs/started.html) if you want to log.


## References
- DDPG: [T. P. Lillicrap et al., Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.](https://arxiv.org/pdf/1509.02971.pdf)
- PPO: [J. Schulman et al., Proximal Policy Optimization Algorithms, arXiv preprint arXiv:1707.06347, 2017.](https://arxiv.org/abs/1707.06347.pdf)
- TD3: [S. Fujimoto et al., Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018.](https://arxiv.org/pdf/1802.09477.pdf)
- SAC: 
    - [T.  Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. arXiv preprint arXiv:1801.01290, 2018.](https://arxiv.org/pdf/1801.01290.pdf)  
    - [T. Haarnoja et al., Soft Actor-Critic Algorithms and Applications. arXiv preprint arXiv:1812.05905, 2018.](https://arxiv.org/pdf/1812.05905.pdf)
- PER: [T. Schaul et al., Prioritized Experience Replay. arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
- HER: [M. Andrychowicz et al., Hindsight Experience Replay. arXiv preprint arXiv:1707.01495, 2017.](https://arxiv.org/pdf/1707.01495.pdf)
- Behavior Cloning: [A. Nair et al., Overcoming Exploration in Reinforcement Learning with Demonstrations. arXiv preprint arXiv:1709.10089, 2017.](https://arxiv.org/pdf/1709.10089.pdf)
- DDPGfD: [M. Vecerik et al., Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards. arXiv preprint arXiv:1707.08817, 2017](https://arxiv.org/pdf/1707.08817.pdf)
