import torch


def compute_gail_reward(discriminator_score: torch.Tensor):
    """Compute gail(imitation) reward of data generated by policy."""
    return (
        -torch.log(torch.sigmoid(discriminator_score) + 1e-8).detach().cpu().numpy()[0]
    )