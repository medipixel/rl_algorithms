import torch
import torch.nn as nn

from rl_algorithms.common.networks.backbones.cnn import CNN
from rl_algorithms.common.networks.backbones.resnet import ResNet
from rl_algorithms.common.networks.base_network import calculate_fc_input_size
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

test_state_dim = (3, 256, 256)


def test_fc_size_calculator():
    calculated_fc_size = calculate_fc_input_size(test_state_dim, cnn_cfg)
    assert calculated_fc_size == 7744


def test_cnn_with_config():
    conv_layer_size = [[1, 32, 64, 64], [1, 32, 21, 21], [1, 64, 11, 11]]
    # test_cnn_model = build_backbone(test_backbone_cfg_params)
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
    test_fc_size_calculator()
    test_cnn_with_config()
