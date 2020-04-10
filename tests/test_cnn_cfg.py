import torch
import torch.nn as nn

from rl_algorithms.common.networks.cnn import CNN, CNNLayer
from rl_algorithms.common.networks.mlp import MLP
from rl_algorithms.common.networks.resnet import BasicBlock, ResNet
from rl_algorithms.dqn.utils import calculate_fc_input_size
from rl_algorithms.utils.config import ConfigDict

network_cfg = ConfigDict(
    dict(
        use_resnet=False,
        cnn_cfg=dict(
            input_sizes=[3, 32, 32],
            output_sizes=[32, 32, 64],
            kernel_sizes=[5, 3, 3],
            strides=[4, 3, 2],
            paddings=[2, 0, 1],
        ),
        resnet_cfg=dict(
            use_bottleneck=False,
            num_blocks=[1, 1, 1, 1],
            block_output_sizes=[32, 32, 64, 64],
            block_strides=[1, 2, 2, 2],
            first_input_size=3,
            first_output_size=32,
            expansion=4,
        ),
    )
)

test_state_dim = (3, 256, 256)


def test_fc_size_calculator():
    calculated_fc_size = calculate_fc_input_size(test_state_dim, network_cfg)
    assert calculated_fc_size == 7744
    resnet_network_cfg = network_cfg.copy()
    resnet_network_cfg.use_resnet = True
    calculate_fc_size = calculate_fc_input_size(test_state_dim, resnet_network_cfg)
    assert calculate_fc_size == 16384


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
    test_mlp = MLP(1, 1, [1])
    resnet_cfg = network_cfg.resnet_cfg
    test_resnet_model = ResNet(
        block=BasicBlock, resnet_cfg=resnet_cfg, fc_layers=test_mlp,
    )
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


def test_cnn_with_config():
    conv_layer_size = [[1, 32, 64, 64], [1, 32, 21, 21], [1, 64, 11, 11]]
    test_mlp = MLP(1, 1, [1])
    cnn_cfg = network_cfg.cnn_cfg
    test_cnn_model = CNN(
        cnn_layers=list(map(CNNLayer, *cnn_cfg.values())), fc_layers=test_mlp
    )
    conv_layers = [
        module for module in test_cnn_model.modules() if isinstance(module, nn.Conv2d)
    ]
    x = torch.zeros(test_state_dim).unsqueeze(0)
    for i, layer in enumerate(conv_layers):
        layer_output = layer(x)
        x = layer_output
        assert list(x.shape) == conv_layer_size[i]


if __name__ == "__main__":
    test_resnet_with_config()
    test_fc_size_calculator()
    test_cnn_with_config()
