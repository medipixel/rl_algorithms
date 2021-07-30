import numpy as np
import torch


def gaifo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    next_states: torch.Tensor,
):
    """Yield mini-batches."""
    batch_size = states.size(0)
    for ep in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], values[
                rand_ids, :
            ], log_probs[rand_ids, :], returns[rand_ids, :], advantages[
                rand_ids, :
            ], next_states[
                rand_ids, :
            ], ep
