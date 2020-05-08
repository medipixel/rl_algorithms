import torch
import torch.nn as nn

from rl_algorithms.common.helper_functions import identity
from rl_algorithms.common.networks.backbones import CNN, ResNet
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict

cnn_cfg = ConfigDict(
    type="CNN",
    configs=dict(
        input_sizes=[3, 32, 32],
        output_sizes=[32, 32, 64],
        kernel_sizes=[5, 3, 3],
        strides=[4, 3, 2],
        paddings=[2, 0, 1],
    ),
)

resnet_cfg = ConfigDict(
    type="ResNet",
    configs=dict(
        use_bottleneck=False,
        num_blocks=[1, 1, 1, 1],
        block_output_sizes=[32, 32, 64, 64],
        block_strides=[1, 2, 2, 2],
        first_input_size=3,
        first_output_size=32,
        expansion=4,
        channel_compression=4,
    ),
)

head_cfg = ConfigDict(
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
)

test_state_dim = (3, 256, 256)


def test_brain():
    """Test wheter brain make fc layer based on backbone's output size."""

    head_cfg.configs.state_size = test_state_dim
    head_cfg.configs.output_size = 8

    model = Brain(resnet_cfg, head_cfg)
    assert model.head.input_size == 16384


def test_cnn_with_config():
    """Test whether CNN module can make proper model according to the configs given."""
    conv_layer_size = [[1, 32, 64, 64], [1, 32, 21, 21], [1, 64, 11, 11]]
    test_cnn_model = CNN(configs=cnn_cfg.configs)
    conv_layers = [
        module for module in test_cnn_model.modules() if isinstance(module, nn.Conv2d)
    ]
    x = torch.zeros(test_state_dim).unsqueeze(0)
    for i, layer in enumerate(conv_layers):
        layer_output = layer(x)
        x = layer_output
        assert list(x.shape) == conv_layer_size[i]


def test_resnet_with_config():
    """Test whether ResNet module can make proper model according to the configs given."""
    conv_layer_size = [
        [1, 32, 256, 256],
        [1, 32, 256, 256],
        [1, 128, 256, 256],
        [1, 128, 256, 256],
        [1, 32, 128, 128],
        [1, 128, 128, 128],
        [1, 128, 128, 128],
        [1, 64, 64, 64],
        [1, 256, 64, 64],
        [1, 256, 64, 64],
        [1, 64, 32, 32],
        [1, 256, 32, 32],
        [1, 256, 32, 32],
        [1, 16, 32, 32],
    ]
    test_resnet_model = ResNet(configs=resnet_cfg.configs)
    conv_layers = [
        module
        for module in test_resnet_model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    x = torch.zeros(test_state_dim).unsqueeze(0)
    skip_x = x
    for i, layer in enumerate(conv_layers):
        if i % 3 == 0:
            layer_output = layer(skip_x)
            skip_x = layer_output
            x = layer_output
        else:
            layer_output = layer(x)
            x = layer_output
        assert list(x.shape) == conv_layer_size[i]


if __name__ == "__main__":
    test_brain()
    test_cnn_with_config()
