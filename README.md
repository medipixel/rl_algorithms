# Reinforcement Learning by Examples
![Build Status](https://travis-ci.org/medipixel/rl_algorithms.svg?branch=master)  

RL key algorithms with [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/).

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
- [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/)
- [Reacher-v2](https://gym.openai.com/envs/Reacher-v2/)

### Prerequisites
Mujoco environment contained `Reacher-v2` require a [Mujoco](https://www.roboti.us/license.html) license.

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
If you check the other options to run file you should command `python run_lunarlander_continuous_v2.py -h`  
If you change **hyper parameters** any algorithms, check the `examples/<env-name>/<algorithm-name>`.
You can make new example python file(e.g. `ddpg-custom`), and you can run
```
python run_lunarlander_continuous_v2.py --algo ddpg-custom
```
You can see to run the algorithm with customized hyper parameters for that you want.

### Development setup
Describe how to install all development dependencies and how to run an automated check of python code.
```
make dev
```

## References
- DDPG: https://arxiv.org/pdf/1509.02971.pdf
- PPO: https://arxiv.org/abs/1707.06347
- TD3: https://arxiv.org/pdf/1802.09477.pdf
- SAC: https://arxiv.org/pdf/1801.01290.pdf
       https://arxiv.org/pdf/1812.05905.pdf
- PER: https://arxiv.org/pdf/1511.05952.pdf
- HER: https://arxiv.org/pdf/1707.01495.pdf
- Behavior Cloning: https://arxiv.org/pdf/1709.10089.pdf
- DDPGfD: https://arxiv.org/pdf/1707.08817.pdf