import torch
import torch.nn as nn

from rl_algorithms.common.networks.base_network import calculate_fc_input_size
from rl_algorithms.common.networks.cnn import CNN
from rl_algorithms.utils.config import ConfigDict

cnn_cfg = ConfigDict(
    configs=dict(
        input_sizes=[3, 32, 32],
        output_sizes=[32, 32, 64],
        kernel_sizes=[5, 3, 3],
        strides=[4, 3, 2],
        paddings=[2, 0, 1],
    )
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


if __name__ == "__main__":
    test_fc_size_calculator()
    test_cnn_with_config()
