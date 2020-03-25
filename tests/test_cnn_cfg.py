from rl_algorithms.dqn.utils import calculate_fc_input_size
from rl_algorithms.utils.config import ConfigDict

# from rl_algorithms.common.networks.cnn import CNN, CNNLayer


def test_fc_size_calculator():
    cnn_cfg = ConfigDict(
        dict(
            input_sizes=[3, 32, 32],
            output_sizes=[32, 32, 64],
            kernel_sizes=[5, 3, 3],
            strides=[4, 3, 2],
            paddings=[2, 0, 1],
        )
    )
    test_state_dim = (3, 256, 256)
    calculated_fc_size = calculate_fc_input_size(test_state_dim, cnn_cfg)
    assert calculated_fc_size == 7744


if __name__ == "__main__":
    test_fc_size_calculator()
