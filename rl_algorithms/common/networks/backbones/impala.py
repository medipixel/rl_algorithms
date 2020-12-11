import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_algorithms.registry import BACKBONES
from rl_algorithms.utils.config import ConfigDict


# pylint: disable=abstract-method
class ImpalaResidual(nn.Module):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, depth):
        super(ImpalaResidual, self).__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


# pylint: disable=abstract-method
class ImpalaCNN(nn.Module):
    """
    The CNN architecture used in the IMPALA paper.
    See https://arxiv.org/abs/1802.01561.
    """

    def __init__(self, image_size, depth_in):
        super(ImpalaCNN, self).__init__()
        layers = []
        for depth_out in [16, 32, 32]:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, 3, padding=1),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    ImpalaResidual(depth_out),
                    ImpalaResidual(depth_out),
                ]
            )
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, 256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


# pylint: disable=abstract-method
@BACKBONES.register_module
class ImpalaModel(nn.Module):
    """
    A Model that computes the base outputs for a CNN that
    also takes state history into account.
    """

    def __init__(self, config: ConfigDict):
        super(ImpalaModel, self).__init__()
        self.cnn = ImpalaCNN(config.IMAGE_SIZE, config.IMAGE_DEPTH)
        self.state_mlp = nn.Sequential(
            nn.Linear(config.STATE_STACK * config.STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.state_mixer = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
        )

    def forward(self, env_info: tuple):
        states = env_info[0]
        observations = env_info[1]
        float_obs = observations.float() / 255.0
        cnn_out = self.cnn(float_obs)
        flat_states = states.view(states.shape[0], -1)
        states_out = self.state_mlp(flat_states.float())
        concatenated = torch.cat([cnn_out, states_out], dim=-1)
        output = self.state_mixer(concatenated)
        return output

    def _base_outs(self, rollout):
        # Even though we could run the entire batch of
        # rollouts in one forward pass, doing so may use
        # too much memory, so we split the rollout up into
        # mini-batches.
        batch_size = 128

        def index_samples():
            for t in range(rollout.num_steps + 1):
                for b in range(rollout.batch_size):
                    yield (t, b)

        def index_batches():
            batch = []
            for x in index_samples():
                batch.append(x)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        result = np.zeros(
            [rollout.num_steps + 1, rollout.batch_size, 256], dtype=np.float32
        )
        for batch in index_batches():
            images = np.array([rollout.obses[t, b] for t, b in batch])
            float_obs = self.tensor(images).float() / 255.0
            cnn_out = self.cnn(float_obs)
            states = self.tensor(np.array([rollout.states[t, b] for t, b in batch]))
            flat_states = states.view(states.shape[0], -1)
            states_out = self.state_mlp(flat_states)
            concatenated = torch.cat([cnn_out, states_out], dim=-1)
            mixed = self.state_mixer(concatenated).detach().cpu().numpy()
            for (t, b), base_out in zip(batch, mixed):
                result[t, b] = base_out

        return result
