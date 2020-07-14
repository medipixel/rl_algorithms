<p align="center">
<img src="https://user-images.githubusercontent.com/17582508/52845370-4a930200-314a-11e9-9889-e00007043872.jpg" align="center">

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/medipixel/rl_algorithms.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/medipixel/rl_algorithms/context:python)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

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
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Curt-Park"><img src="https://avatars3.githubusercontent.com/u/14961526?v=4" width="100px;" alt=""/><br /><sub><b>Jinwoo Park (Curt)</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=Curt-Park" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/MrSyee"><img src="https://avatars3.githubusercontent.com/u/17582508?v=4" width="100px;" alt=""/><br /><sub><b>Kyunghwan Kim</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=MrSyee" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/darthegg"><img src="https://avatars3.githubusercontent.com/u/16010242?v=4" width="100px;" alt=""/><br /><sub><b>darthegg</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=darthegg" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/mclearning2"><img src="https://avatars3.githubusercontent.com/u/43226417?v=4" width="100px;" alt=""/><br /><sub><b>Mincheol Kim</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=mclearning2" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/minseop4898"><img src="https://avatars1.githubusercontent.com/u/34338299?v=4" width="100px;" alt=""/><br /><sub><b>김민섭</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=minseop4898" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jinPrelude"><img src="https://avatars1.githubusercontent.com/u/16518993?v=4" width="100px;" alt=""/><br /><sub><b>Leejin Jung</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=jinPrelude" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/cyoon1729"><img src="https://avatars2.githubusercontent.com/u/33583101?v=4" width="100px;" alt=""/><br /><sub><b>Chris Yoon</b></sub></a><br /><a href="https://github.com/medipixel/rl_algorithms/commits?author=cyoon1729" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.

## Algorithms

0. [Advantage Actor-Critic (A2C)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/a2c)
1. [Deep Deterministic Policy Gradient (DDPG)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/ddpg)
2. [Proximal Policy Optimization Algorithms (PPO)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/ppo)
3. [Twin Delayed Deep Deterministic Policy Gradient Algorithm (TD3)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/td3)
4. [Soft Actor Critic Algorithm (SAC)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/sac)
5. [Behaviour Cloning (BC with DDPG, SAC)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/bc)
6. [From Demonstrations (DDPGfD, SACfD, DQfD)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/fd)
7. [Rainbow DQN](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/dqn)
8. [Rainbow IQN (without DuelingNet)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/dqn) - DuelingNet [degrades performance](https://github.com/medipixel/rl_algorithms/pull/137)
9. Rainbow IQN (with [ResNet](https://github.com/medipixel/rl_algorithms/blob/master/rl_algorithms/common/networks/backbones/resnet.py))
10. [Recurrent Replay DQN (R2D1)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/recurrent)
11. [Distributed Pioritized Experience Replay (Ape-X)](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/common/distributed)
12. [Policy Distillation](https://github.com/medipixel/rl_algorithms/tree/master/rl_algorithms/distillation)

## Performance

We have tested each algorithm on some of the following environments.
- [PongNoFrameskip-v4](https://github.com/medipixel/rl_algorithms/tree/master/configs/pong_no_frameskip_v4)
- [LunarLanderContinuous-v2](https://github.com/medipixel/rl_algorithms/tree/master/configs/lunarlander_continuous_v2)
- [LunarLander_v2](https://github.com/medipixel/rl_algorithms/tree/master/configs/lunarlander_v2)
- [Reacher-v2](https://github.com/medipixel/rl_algorithms/tree/master/configs/reacher-v2)

❗Please note that this won't be frequently updated.


#### PongNoFrameskip-v4

**RainbowIQN** learns the game incredibly fast! It accomplishes the perfect score (21) [within 100 episodes](https://app.wandb.ai/curt-park/dqn/runs/b2p9e9f7/logs)!
The idea of RainbowIQN is roughly suggested from [W. Dabney et al.](https://arxiv.org/pdf/1806.06923.pdf).

See [W&B Log](https://app.wandb.ai/curt-park/dqn/reports?view=curt-park%2FPong%20%28DQN%20%2F%20C51%20%2F%20IQN%20%2F%20IQN%20-double%20q%29) for more details. (The performance is measured on the commit [4248057](https://github.com/medipixel/rl_algorithms/pull/158))

![pong_dqn](https://user-images.githubusercontent.com/17582508/56282434-1e93fd00-614a-11e9-9c31-af32e119d5b6.png)

**RainbowIQN with ResNet**'s performance and learning speed were similar to those of RainbowIQN. Also we confirmed that **R2D1 (w/ Dueling, PER)** converges well in the Pong enviornment, though not as fast as RainbowIQN (in terms of update step).

Although we were only able to test **Ape-X DQN (w/ Dueling)** with 4 workers due to limitations to computing power, we observed a significant speed-up in carrying out update steps (with batch size 512). Ape-X DQN learns Pong game in about 2 hours, compared to 4 hours for serial Dueling DQN.

See [W&B Log](https://app.wandb.ai/medipixel_rl/PongNoFrameskip-v4/reports/200626-integration-test--VmlldzoxNTE1NjE) for more details. (The performance is measured on the commit [9e897ad](https://github.com/medipixel/rl_algorithms/commit/9e897adfe93600c1db85ce1a7e064064b025c2c3))
![pong dqn with resnet & rnn](https://user-images.githubusercontent.com/17582508/85813189-80fc7a80-b79d-11ea-96cf-947a62e380f3.png)

![apex dqn](https://user-images.githubusercontent.com/17582508/85814263-83ac9f00-b7a0-11ea-9cdc-ff29de9a6d54.png)

#### LunarLander-v2 / LunarLanderContinuous-v2

We used these environments just for a quick verification of each algorithm, so some of experiments may not show the best performance. 

##### 👇 Click the following lines to see the figures.
<details><summary><b>LunarLander-v2: RainbowDQN, RainbowDQfD, R2D1 </b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/LunarLander-v2/reports/200626-integration-test--VmlldzoxNTE2MzA">W&B log</a> for more details. (The performance is measured on the commit <a href="https://github.com/medipixel/rl_algorithms/commit/9e897adfe93600c1db85ce1a7e064064b025c2c3">9e897ad</a>)

![lunarlander-v2_dqn](https://user-images.githubusercontent.com/17582508/85815561-a5f3ec00-b7a3-11ea-8d7c-8d54953d0c07.png)
</p>
</details>

<details><summary><b>LunarLanderContinuous-v2: A2C, PPO, DDPG, TD3, SAC</b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/LunarLanderContinuous-v2/reports/200626-integration-test--VmlldzoxNDg1MjU">W&B log</a> for more details. (The performance is measured on the commit <a href="https://github.com/medipixel/rl_algorithms/commit/9e897adfe93600c1db85ce1a7e064064b025c2c3">9e897ad</a>)

![lunarlandercontinuous-v2_baselines](https://user-images.githubusercontent.com/17582508/85818298-43065300-b7ab-11ea-9ee0-1eda855498ed.png)
</p>
</details>

<details><summary><b>LunarLanderContinuous-v2: DDPG, DDPGfD, BC-DDPG</b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/LunarLanderContinuous-v2/reports/200626-integration-test--VmlldzoxNDg1MjU">W&B log</a> for more details. (The performance is measured on the commit <a href="https://github.com/medipixel/rl_algorithms/commit/9e897adfe93600c1db85ce1a7e064064b025c2c3">9e897ad</a>)

![lunarlandercontinuous-v2_ddpg](https://user-images.githubusercontent.com/17582508/85818519-c9bb3000-b7ab-11ea-9473-08476a959a0c.png)
</p>
</details>

<details><summary><b>LunarLanderContinuous-v2: SAC, SACfD, BC-SAC</b></summary>
<p><br>
See <a href="https://app.wandb.ai/medipixel_rl/LunarLanderContinuous-v2/reports/200626-integration-test--VmlldzoxNDg1MjU">W&B log</a> for more details. (The performance is measured on the commit <a href="https://github.com/medipixel/rl_algorithms/commit/9e897adfe93600c1db85ce1a7e064064b025c2c3">9e897ad</a>)

![lunarlandercontinuous-v2_sac](https://user-images.githubusercontent.com/17582508/85818654-1acb2400-b7ac-11ea-8641-d559839cab62.png)
</p>
</details>

#### Reacher-v2

We reproduced the performance of **DDPG**, **TD3**, and **SAC** on Reacher-v2 (Mujoco). They reach the score around -3.5 to -4.5.

##### 👇 Click the following the line to see the figures.

<details><summary><b>Reacher-v2: DDPG, TD3, SAC</b></summary>
<p><br>

See [W&B Log](https://app.wandb.ai/medipixel_rl/reacher-v2/reports?view=curt-park%2FBaselines%20%23158) for more details.

![reacher-v2_baselines](https://user-images.githubusercontent.com/17582508/56282421-163bc200-614a-11e9-8d4d-2bb520575fbb.png)

</p>
</details>


## Getting started

#### Prerequisites
* This repository is tested on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.6.1+
    ```
    $ conda create -n rl_algorithms python=3.6.9
    $ conda activate rl_algorithms
    ```
* In order to run Mujoco environments (e.g. `Reacher-v2`), you need to acquire [Mujoco license](https://www.roboti.us/license.html).

#### Installation
First, clone the repository.
```
git clone https://github.com/medipixel/rl_algorithms.git
cd rl_algorithms
```

###### For users
Install packages required to execute the code. It includes `python setup.py install`. Just type:
```
make dep
```

###### For developers
If you want to modify code you should configure formatting and linting settings. It automatically runs formatting and linting when you commit the code. Contrary to `make dep` command, it includes `python setup.py develop`. Just type:

```
make dev
```

After having done `make dev`, you can validate the code by the following commands.
```
make format  # for formatting
make test  # for linting
```

#### Usages
You can train or test `algorithm` on `env_name` if `configs/env_name/algorithm.py` exists. (`configs/env_name/algorithm.py` contains hyper-parameters)
```
python run_env_name.py --cfg-path <config-path>
``` 

e.g. running soft actor-critic on LunarLanderContinuous-v2.
```
python run_lunarlander_continuous_v2.py --cfg-path ./configs/lunarlander_continuous_v2/sac.py <other-options>
```

e.g. running a custom agent, **if you have written your own configs**: `configs/env_name/ddpg-custom.py`.
```
python run_env_name.py --cfg-path ./configs/lunarlander_continuous_v2/ddpg-custom.py
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

#### Arguments for distributed training in run-files
- `--max-episode-steps <int>`
    - Set maximum update step for learner as a stopping criterion for training loop. If the number is less than or equal to 0, it uses the default maximum step number of the environment.
- `--off-worker-render`
    - Turn off rendering of individual workers.
- `--off-logger-render`
    - Turn off rendering of logger tests.
- `--worker-verbose`
    - Turn on printing episode run info for individual workers 
    

#### Show feature map with Grad-CAM
You can show a feature map that the trained agent extract using **[Grad-CAM(Gradient-weighted Class Activation Mapping)](https://arxiv.org/pdf/1610.02391.pdf)**. Grad-CAM is a way of combining feature maps using the gradient signal, and produce a coarse localization map of the important regions in the image. You can use it by adding [Grad-CAM config](https://github.com/medipixel/rl_algorithms/blob/master/configs/pong_no_frameskip_v4/dqn.py#L39) and `--grad-cam` flag when you run. For example:
```
python run_env_name.py --cfg-path <config-path> --test --grad-cam
```
It can be only used the agent that uses convolutional layers like **DQN for Pong environment**. You can see feature maps of all the configured convolution layers.

<img src="https://user-images.githubusercontent.com/17582508/79204132-02b75a00-7e77-11ea-9c78-ab543055bd4f.gif" width="400" height="400" align="center"/>

#### Using policy distillation
You can use policy distillation if you have checkpoints of a learned agent.

First, collect the data in the desired directory(`distillation-buffer-path`) with the learned teacher agent:
```
python run_env_name.py --test --load-from <teacher-checkpoint-path> --distillation-buffer-path <path-to-store-data> --cfg-path <distillation-config-path>
```
When you do this, the model structure of **distillation config file** should be the same as the teacher. You can set the number of data to be stored by the `buffer_size` variable in the distillation config file.

Second, you can train the student model with the following command:
```
python run_env_name.py --distillation-buffer-path <path-where-data-is-stored> --cfg-path <distillation-config-path>
```
You can set `epoch` and `batch_size` of the student learning through `epochs` and `batch_size` variables in the distillation config file. The checkpoint file of the student will be saved in `./checkpoint/env_name/DistillationDQN/`.

Finally, You can test performance in the same way as **the original agent** using the checkpoint file of the student:
```
python run_env_name.py --test --load-from <student-checkpoint-path> --cfg-path <config-path>
```
You **must use the original agent config file** with the same model structure as the student, not the distillation config file. (e.g. `distillation_dqn.py` -> `dqn.py`)

#### W&B for logging
We use [W&B](https://www.wandb.com/) for logging of network parameters and others. For logging, please follow the steps below after requirement installation:

>0. Create a [wandb](https://www.wandb.com/) account
>1. Check your **API key** in settings, and login wandb on your terminal: `$ wandb login API_KEY`
>2. Initialize wandb: `$ wandb init`

For more details, read [W&B tutorial](https://docs.wandb.com/docs/started.html).

## Class Diagram
Class diagram at [#135](https://github.com/medipixel/rl_algorithms/pull/135).

❗This won't be frequently updated.

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
17. [Ramprasaath R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." arXiv preprint arXiv:1610.02391, 2016.](https://arxiv.org/pdf/1610.02391.pdf)
18. [Kaiming He et al., "Deep Residual Learning for Image Recognition." arXiv preprint arXiv:1512.03385, 2015.](https://arxiv.org/pdf/1512.03385)
19. [Steven Kapturowski et al., "Recurrent Experience Replay in Distributed Reinforcement Learning." in International Conference on Learning Representations https://openreview.net/forum?id=r1lyTjAqYX, 2019.](https://openreview.net/forum?id=r1lyTjAqYX)
20. [Horgan et al., "Distributed Prioritized Experience Replay." in International Conference on Learning Representations, 2018](https://arxiv.org/pdf/1803.00933.pdf)