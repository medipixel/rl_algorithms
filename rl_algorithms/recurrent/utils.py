from typing import Any, Tuple

import torch

from rl_algorithms.common.helper_functions import make_one_hot


def infer_leading_dims(tensor: torch.Tensor, dim: int) -> Tuple[int, int, int, Tuple]:
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)

    Example:
        Let's assume that the shape of the tensor is [20, 32, 4, 84, 84], which
        each dimensions is refer to sequence_size, batch_size, frame_stack, width, height.
        lead_dim = 2, first_dim = 20, second_dim = 32, shape = (4, 84, 84)

    Returns:
        lead_dim (int): --number of leading dims found.
        first_dim (int): --size of first leading dim, if two leading dims, o/w 1.
        second_dim (int): --size of first leading dim if one, second leading dim if two, o/w 1.
        shape (tuple): tensor shape after leading dims.

    Reference:
        https://github.com/astooke/rlpyt/blob/master/rlpyt/models/dqn/atari_r2d1_model.py
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        first_dim, second_dim = tensor.shape[:2]
    else:
        first_dim = 1
        second_dim = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, first_dim, second_dim, shape


def restore_leading_dims(
    tensors: torch.Tensor, lead_dim: int, first_dim: int = 1, second_dim: int = 1
) -> torch.Tensor:
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [second_dim], or [first_dim, second_dim].
    Assumes input tensors already have a leading Batch dimension, which might need
    to be removed. (Typically the last layer of model will compute with leading batch
    dimension.)  For use in model ``forward()`` method, so that output dimensions
    match input dimensions, and the same model can be used for any such case.
    Use with outputs from ``infer_leading_dims()``.

    Reference:
        https://github.com/astooke/rlpyt/blob/master/rlpyt/models/dqn/atari_r2d1_model.py
    """
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put first_dim.)
        tensors = tuple(t.view((first_dim, second_dim) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove second_dim=1 dim.)
        assert second_dim == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]


def valid_from_done(done: torch.Tensor) -> torch.Tensor:
    """Returns a float mask which is zero for all time-steps after a
    `done=True` is signaled.  This function operates on the leading dimension
    of `done`, assumed to correspond to time [T,...], other dimensions are
    preserved.

    Reference: https://github.com/astooke/rlpyt/blob/master/rlpyt/algos/utils.py
    """
    done = done.type(torch.float).squeeze()
    valid = torch.ones_like(done)
    valid[1:] = 1 - torch.clamp(torch.cumsum(done[:-1], dim=0), max=1)
    valid = valid[-1] == 0
    return valid


def slice_r2d1_arguments(
    experiences: Tuple[Any, ...],
    burn_in_step: int,
    output_size: int,
) -> tuple:
    """Get mini-batch sequence-size transitions and slice
    in accordance with R2D1 agent loss calculating process.
    return tuples bound by target relationship.

    Example:
        state_tuple = (state, target_state)
    """
    states, actions, rewards, hiddens, dones = experiences[:5]

    burnin_states = states[:, 1:burn_in_step]
    target_burnin_states = states[:, 1 : burn_in_step + 1]
    agent_states = states[:, burn_in_step:-1]
    target_states = states[:, burn_in_step + 1 :]

    burnin_prev_actions = make_one_hot(
        actions[:, : burn_in_step - 1],
        output_size,
    )
    target_burnin_prev_actions = make_one_hot(actions[:, :burn_in_step], output_size)
    agent_actions = actions[:, burn_in_step:-1].long().unsqueeze(-1)
    prev_actions = make_one_hot(
        actions[:, burn_in_step - 1 : -2],
        output_size,
    )
    target_prev_actions = make_one_hot(
        actions[:, burn_in_step:-1].long(),
        output_size,
    )

    burnin_prev_rewards = rewards[:, : burn_in_step - 1].unsqueeze(-1)
    target_burnin_prev_rewards = rewards[:, :burn_in_step].unsqueeze(-1)
    agent_rewards = rewards[:, burn_in_step:-1].unsqueeze(-1)
    prev_rewards = rewards[:, burn_in_step - 1 : -2].unsqueeze(-1)
    target_prev_rewards = agent_rewards
    burnin_dones = dones[:, 1:burn_in_step].unsqueeze(-1)
    burnin_target_dones = dones[:, 1 : burn_in_step + 1].unsqueeze(-1)
    agent_dones = dones[:, burn_in_step:-1].unsqueeze(-1)
    init_rnn_state = hiddens[:, 0].squeeze(1).contiguous()

    burnin_state_tuple = (burnin_states, target_burnin_states)
    state_tuple = (agent_states, target_states)
    burnin_prev_action_tuple = (burnin_prev_actions, target_burnin_prev_actions)
    prev_action_tuple = (prev_actions, target_prev_actions)
    burnin_prev_reward_tuple = (burnin_prev_rewards, target_burnin_prev_rewards)
    prev_reward_tuple = (prev_rewards, target_prev_rewards)
    burnin_dones_tuple = (burnin_dones, burnin_target_dones)

    return (
        burnin_state_tuple,
        state_tuple,
        burnin_prev_action_tuple,
        agent_actions,
        prev_action_tuple,
        burnin_prev_reward_tuple,
        agent_rewards,
        prev_reward_tuple,
        burnin_dones_tuple,
        agent_dones,
        init_rnn_state,
    )
