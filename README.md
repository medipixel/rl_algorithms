<p align="center">

<img src="https://user-images.githubusercontent.com/17582508/52845370-4a930200-314a-11e9-9889-e00007043872.jpg" align="center">

![Build Status](https://travis-ci.org/medipixel/rl_algorithms.svg?branch=master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/medipixel/rl_algorithms.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/medipixel/rl_algorithms/context:python)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors)

</p>

## Contents

* [Welcome!](https://github.com/medipixel/rl_algorithms#welcome)
* [Contributors](https://github.com/medipixel/rl_algorithms#contributors)
* [Algorithms](https://github.com/medipixel/rl_algorithms#algorithms)
* [Performance](https://github.com/medipixel/rl_algorithms#performance)
* [Getting Started](https://github.com/medipixel/rl_algorithms#getting-started)
* [Class Diagram](https://github.com/medipixel/rl_algorithms#class-diagram)
* [References](https://github.com/medipixel/rl_algorithms#references)


## Welcome!
This repository contains Reinforcement Learning algorithms which are being used for research activities at Medipixel. The source code will be frequently updated. 
We are warmly welcoming external contributors! :)

|<img src="https://user-images.githubusercontent.com/17582508/52840582-18c76e80-313d-11e9-9752-3d6138f39a15.gif" width="260" height="180"/>|<img src="https://media.giphy.com/media/ZxLNajigOcLyeUnOwg/giphy.gif" width="160" height="180"/>|<img src="https://media.giphy.com/media/1mikGEln2lArKMQ6Pt/giphy.gif" width="260" height="180"/>|
|:---:|:---:|:---:|
|BC agent on LunarLanderContinuous-v2|RainbowIQN agent on PongNoFrameskip-v4|SAC agent on Reacher-v2|

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Curt-Park"><img src="https://avatars3.githubusercontent.com/u/14961526?v=4" width="100px;" alt="Jinwoo Park (Curt)"/><br /><sub><b>Jinwoo Park (Curt)</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=Curt-Park" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/MrSyee"><img src="https://avatars3.githubusercontent.com/u/17582508?v=4" width="100px;" alt="Kyunghwan Kim"/><br /><sub><b>Kyunghwan Kim</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=MrSyee" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/darthegg"><img src="https://avatars0.githubusercontent.com/u/16010242?v=4" width="100px;" alt="darthegg"/><br /><sub><b>darthegg</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=darthegg" title="Code">💻</a></td>
    <td align="center"><a href="https://mclearninglab.tistory.com"><img src="https://avatars0.githubusercontent.com/u/43226417?v=4" width="100px;" alt="Mincheol Kim"/><br /><sub><b>Mincheol Kim</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=mclearning2" title="Code">💻</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.

## Algorithms

0. [Advantage Actor-Critic (A2C)](https://github.com/medipixel/rl_algorithms/blob/master/rl_algorithms/a2c)
1. [Deep Deterministic Policy Gradient (DDPG)](https://github.com/medipixel/rl_algorithms/blob/master/rl_algorithms/ddpg)
2. [Proximal Policy Optimization Algorithms (PPO)](https://github.com/medipixel/rl_algorithms/blob/master/rl_algorithms/ppo)
3. [Twin Delayed Deep Deterministic Policy Gradient Algorithm (TD3)](https://github.com/medipixel/rl_algorithms/blob/master/rl_algorithms/td3)
4. [Soft Actor Critic Algorithm (SAC)](https://github.com/medipixel/rl_algorithms/blob/master/rl_algorithms/sac/agent.py)
5. [Behaviour Cloning (BC with DDPG, SAC)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/bc)
6. [Prioritized Experience Replay (PER with DDPG)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/per)
7. [From Demonstrations (DDPGfD, SACfD, DQfD)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/fd)
8. [Rainbow DQN](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/dqn)
9. [Rainbow IQN (without DuelingNet)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/dqn) - DuelingNet [degrades performance](https://github.com/medipixel/rl_algorithms/pull/137)


## Performance

We have tested each algorithm on some of the following environments.
- [LunarLanderContinuous-v2](https://github.com/medipixel/rl_algorithms/tree/master/examples/lunarlander_continuous_v2)
- [LunarLander_v2](https://github.com/medipixel/rl_algorithms/tree/master/examples/lunarlander_v2)
- [Reacher-v2](https://github.com/medipixel/rl_algorithms/tree/master/examples/reacher-v2)
- [PongNoFrameskip-v4](https://github.com/medipixel/rl_algorithms/tree/master/examples/pong_no_frameskip_v4)

The performance is measured on the commit [4248057](https://github.com/medipixel/rl_algorithms/pull/158). Please note that this won't be frequently updated.


#### Reacher-v2

We reproduced the performance of **DDPG**, **TD3**, and **SAC** on Reacher-v2 (Mujoco). They reach the score around -3.5 to -4.5.
See [W&B Log](https://app.wandb.ai/medipixel_rl/reacher-v2/reports?view=curt-park%2FBaselines%20%23158) for more details.

![reacher-v2_baselines](https://user-images.githubusercontent.com/17582508/56282421-163bc200-614a-11e9-8d4d-2bb520575fbb.png)

#### PongNoFrameskip-v4

**RainbowIQN** learns the game incredibly fast! It accomplishes the perfect score (21) [within 100 episodes](https://app.wandb.ai/curt-park/dqn/runs/b2p9e9f7/logs)!
The idea of RainbowIQN is roughly suggested from [W. Dabney et al.](https://arxiv.org/pdf/1806.06923.pdf).
See [W&B Log](https://app.wandb.ai/curt-park/dqn/reports?view=curt-park%2FPong%20%28DQN%20%2F%20C51%20%2F%20IQN%20%2F%20IQN%20-double%20q%29) for more details.

![pong_dqn](https://user-images.githubusercontent.com/17582508/56282434-1e93fd00-614a-11e9-9c31-af32e119d5b6.png)

#### LunarLander-v2 / LunarLanderContinuous-v2

We used these environments just for a quick verification of each algorithm, so some of experiments may not show the best performance. Click the following lines to see the figures.

<details><summary><b>LunarLander-v2: RainbowDQN, RainbowDQfD</b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/lunarlander_v2/reports?view=curt-park%2FBaselines%20%23158">W&B log</a> for more details.

![lunarlander-v2_dqn](https://user-images.githubusercontent.com/17582508/56282616-85b1b180-614a-11e9-99a7-b2d9715a6bc6.png)
</p>
</details>

<details><summary><b>LunarLanderContinuous-v2: A2C, PPO, DDPG, TD3, SAC</b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/lunarlander_continuous_v2/reports?view=curt-park%2FBaselines%20%23158">W&B log</a> for more details.

![lunarlandercontinuous-v2_baselines](https://user-images.githubusercontent.com/14961526/56286075-bcd89080-6153-11e9-9ef5-8336bd5e2114.png)
</p>
</details>

<details><summary><b>LunarLanderContinuous-v2: DDPG, PER-DDPG, DDPGfD, BC-DDPG</b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/lunarlander_continuous_v2/reports?view=curt-park%2FDDPG%20%23158">W&B log</a> for more details.

![lunarlandercontinuous-v2_ddpg](https://user-images.githubusercontent.com/17582508/56282350-ea204100-6149-11e9-8642-0b9d6171ad9f.png)
</p>
</details>

<details><summary><b>LunarLanderContinuous-v2: SAC, SACfD, BC-SAC</b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/lunarlander_continuous_v2/reports?view=curt-park%2FSAC%20%23158">W&B log</a> for more details.

![lunarlandercontinuous-v2_sac](https://user-images.githubusercontent.com/17582508/56282450-2c498280-614a-11e9-836d-86fdc240cd17.png)
</p>
</details>

## Getting started

#### Prerequisites
* This repository is tested on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.6.1+
    ```
    $ conda create -n rl_algorithms python=3.6.1
    $ conda activate rl_algorithms
    ```
* In order to run Mujoco environments (e.g. `Reacher-v2`), you need to acquire [Mujoco license](https://www.roboti.us/license.html).

#### Installation
First, clone the repository.
```
git clone https://github.com/medipixel/rl_algorithms.git
cd rl_algorithms
```
Secondly, install packages required to execute the code. Just type:
```
make dep
```

###### For developers
You need to type the additional command which configures formatting and linting settings. It automatically runs formatting and linting when you commit the code.

```
make dev
```

After having done `make dev`, you can validate the code by the following commands.
```
make format  # for formatting
make test  # for linting
```

#### Usages
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

#### Arguments for run-files

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

#### W&B for logging
We use [W&B](https://www.wandb.com/) for logging of network parameters and others. For logging, please follow the steps below after requirement installation:

>0. Create a [wandb](https://www.wandb.com/) account
>1. Check your **API key** in settings, and login wandb on your terminal: `$ wandb login API_KEY`
>2. Initialize wandb: `$ wandb init`

For more details, read [W&B tutorial](https://docs.wandb.com/docs/started.html).

## Class Diagram
Class diagram at [#135](https://github.com/medipixel/rl_algorithms/pull/135).
This won't be frequently updated.
![RL_Algorithms_ClassDiagram](https://user-images.githubusercontent.com/16010242/55934443-812d5a80-5c6b-11e9-9b31-fa8214965a55.png)

## References
0. [T. P. Lillicrap et al., "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971, 2015.](https://arxiv.org/pdf/1509.02971.pdf)
1. [J. Schulman et al., "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017.](https://arxiv.org/abs/1707.06347.pdf)
2. [S. Fujimoto et al., "Addressing function approximation error in actor-critic methods." arXiv preprint arXiv:1802.09477, 2018.](https://arxiv.org/pdf/1802.09477.pdf)
3. [T.  Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." arXiv preprint arXiv:1801.01290, 2018.](https://arxiv.org/pdf/1801.01290.pdf)
4. [T. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications." arXiv preprint arXiv:1812.05905, 2018.](https://arxiv.org/pdf/1812.05905.pdf)
5. [T. Schaul et al., "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
6. [M. Andrychowicz et al., "Hindsight Experience Replay." arXiv preprint arXiv:1707.01495, 2017.](https://arxiv.org/pdf/1707.01495.pdf)
7. [A. Nair et al., "Overcoming Exploration in Reinforcement Learning with Demonstrations." arXiv preprint arXiv:1709.10089, 2017.](https://arxiv.org/pdf/1709.10089.pdf)
8. [M. Vecerik et al., "Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards."arXiv preprint arXiv:1707.08817, 2017](https://arxiv.org/pdf/1707.08817.pdf)
9. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518
(7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
10. [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)
11. [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv preprint arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)
12. [T. Hester et al., "Deep Q-learning from Demonstrations." arXiv preprint arXiv:1704.03732, 2017.](https://arxiv.org/pdf/1704.03732.pdf)
13. [M. G. Bellemare et al., "A Distributional Perspective on Reinforcement Learning." arXiv preprint arXiv:1707.06887, 2017.](https://arxiv.org/pdf/1707.06887.pdf)
14. [M. Fortunato et al., "Noisy Networks for Exploration." arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/pdf/1706.10295.pdf)
15. [M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv preprint arXiv:1710.02298, 2017.](https://arxiv.org/pdf/1710.02298.pdf)
16. [W. Dabney et al., "Implicit Quantile Networks for Distributional Reinforcement Learning." arXiv preprint arXiv:1806.06923, 2018.](https://arxiv.org/pdf/1806.06923.pdf)
