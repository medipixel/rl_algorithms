import torch
from torch.distributions import Distribution, Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TanhNormal(Distribution):
    """Represent distribution of X where X ~ tanh(Z) and Z ~ N(mean, std).

    Note: this is not very numerically stable.

    This is taken from:
        https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/distributions.py
    """

    def __init__(
        self, normal_mean: torch.Tensor, normal_std: torch.Tensor, epsilon: float = 1e-6
    ):
        """Initialization.

        Args:
            normal_mean (torch.Tensor): Mean of the normal distribution
            normal_std (torch.Tensor): Std of the normal distribution
            epsilon (float): Numerical stability epsilon when computing log-prob.
        """
        super(TanhNormal, self).__init__()

        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n: int, return_pre_tanh_value: bool = False):
        """Sample n numbers from the distribution."""
        z = self.normal.sample_n(n)

        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value: torch.Tensor, pre_tanh_value: torch.Tensor = None):
        """Return log probability.

        Args:
            value (torch.Tensor): some value, x
            pre_tanh_value (torch.Tensor): arctanh(x)
        """
        if pre_tanh_value is None:
            pre_tanh_value = self.pre_tanh_value(value)

        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value.pow(2) + self.epsilon
        )

    def sample(self, return_pretanh_value: bool = False):
        """Return sampled values.

        Note:
            Gradients will and should *not* pass through this operation.
            See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z

        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value: bool = False):
        """Return sampled differentiable values."""
        #        z = (
        #            self.normal_mean
        #            + self.normal_std
        #            * Normal(
        #                torch.zeros(self.normal_mean.size()).to(device),
        #                torch.ones(self.normal_std.size()).to(device),
        #            ).sample()
        #        )
        #        z.requires_grad_()

        z = self.normal.rsample()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    @staticmethod
    def pre_tanh_value(value: torch.Tensor) -> torch.Tensor:
        """Return pre-tanh value."""
        return torch.log((1 + value) / (1 - value)) / 2

    @property
    def mean(self) -> torch.Tensor:
        """Return mean."""
        return self.normal_mean

    @property
    def stddev(self) -> torch.Tensor:
        """Return std."""
        return self.normal_std

    @property
    def variance(self) -> torch.Tensor:
        """Return variance."""
        return self.normal_std.pow(2)

    def entropy(self):
        pass

    def cdf(self):
        pass

    def enumerate_support(self):
        pass

    def icdf(self):
        pass
