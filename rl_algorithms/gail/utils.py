import torch


def concat_state_action_tensor(
    state: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Concatenate state tensor and action tensor to make input of discriminator."""
    assert isinstance(state, torch.Tensor)
    assert isinstance(action, torch.Tensor)

    return torch.cat([state, action], dim=-1)


def compute_gail_reward(discriminator_score: torch.Tensor):
    """Compute gail(imitation) reward of data generated by policy.
    Reference:
        https://github.com/openai/baselines/blob/master/baselines/gail/adversary.py
    """
    return (
        -torch.log(1 - torch.sigmoid(discriminator_score) + 1e-8)
        .detach()
        .cpu()
        .numpy()[0]
    )
