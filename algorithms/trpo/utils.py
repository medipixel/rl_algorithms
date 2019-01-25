# -*- coding: utf-8 -*-
"""Utility functions repeatedly used for TRPO.

This module has TRPO util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://arxiv.org/abs/1502.05477
"""

from typing import Callable, Deque

import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

# device selection: cpu / gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def decompose_memory(memory: Deque):
    """Decompose states, actions, rewards, dones from the memory."""
    memory_np: np.ndarray = np.array(memory)

    states = torch.from_numpy(np.vstack(memory_np[:, 0])).float().to(device)
    actions = torch.from_numpy(np.vstack(memory_np[:, 1])).float().to(device)
    rewards = torch.from_numpy(np.vstack(memory_np[:, 2])).float().to(device)
    dones = (
        torch.from_numpy(np.vstack(memory_np[:, 3]).astype(np.uint8)).float().to(device)
    )

    return states, actions, rewards, dones


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
def get_flat_params_from(model: nn.Module) -> torch.Tensor:
    """Return flat_params from the model parameters."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
def set_flat_params_to(model: nn.Module, flat_params: torch.Tensor):
    """Set the model parameters as flat_params."""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind : prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
def get_flat_grad_from(net: nn.Module, grad_grad: bool = False):
    """Return flat grads of the model parameters."""
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
def conjugate_gradients(
    Avp: Callable, b: torch.Tensor, nsteps: int, residual_tol: float = 1e-10
) -> torch.Tensor:
    """Demmel p 312."""
    x = torch.zeros(b.size()).to(device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
def linesearch(
    model: nn.Module,
    f: Callable,
    x: torch.Tensor,
    fullstep: torch.Tensor,
    expected_improve_rate: torch.Tensor,
    max_backtracks: int = 10,
    accept_ratio: float = 0.1,
):
    """Backtracking linesearch.

    where expected_improve_rate is the slope the slope dy/dx
    at the initial point.
    """
    fval = f(True).data
    for (_n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
        xnew = x + torch.Tensor([stepfrac]).to(device) * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, x


def critic_step(
    critic: nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    l2_reg: torch.Tensor,
    lbfgs_max_iter: int,
) -> torch.Tensor:
    """Train critic and return the loss."""
    get_value_loss = ValueLoss(critic, states, targets, l2_reg)
    flat_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss,
        get_flat_params_from(critic).cpu().double().numpy(),
        maxiter=lbfgs_max_iter,
    )
    set_flat_params_to(critic, torch.Tensor(flat_params))

    loss = (critic(states) - targets).pow(2).mean()

    return loss


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
def actor_step(
    old_actor: nn.Module,
    actor: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    max_kl: float,
    damping: float,
) -> torch.Tensor:
    """Train actor and return the loss."""
    _, old_dist = old_actor(states)
    old_log_prob = old_dist.log_prob(actions)

    def get_loss(volatile: bool = False):
        if volatile:
            with torch.no_grad():
                _, dist = actor(states)
        else:
            _, dist = actor(states)

        log_prob = dist.log_prob(actions)
        loss = -advantages * torch.exp(log_prob - old_log_prob)

        return loss.mean()

    def get_kl():
        _, dist = actor(states)

        return kl_divergence(old_dist, dist).sum(1, keepdim=True)

    loss = get_loss()
    grads = torch.autograd.grad(loss, actor.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v: torch.Tensor):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, actor.parameters())
        flat_grad_grad_kl = torch.cat(
            [grad.contiguous().view(-1) for grad in grads]
        ).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]
    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    prev_params = get_flat_params_from(actor)

    _, new_params = linesearch(
        actor, get_loss, prev_params, fullstep, neggdotstepdir / lm[0]
    )

    set_flat_params_to(actor, new_params)

    return loss


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
class ValueLoss:
    """Value Loss calculator.

    Attributes:
        model (nn.Module): model to predict values
        states (torch.Tensor): current states
        targets (torch.Tensor): current returns
        l2_reg (float): weight decay rate

    """

    def __init__(
        self,
        model: nn.Module,
        states: torch.Tensor,
        targets: torch.Tensor,
        l2_reg: float,
    ):
        """Initialization."""
        self.model = model
        self.states = states
        self.targets = targets
        self.l2_reg = l2_reg

    def __call__(self, flat_params: torch.Tensor):
        """Return value loss."""
        set_flat_params_to(self.model, torch.Tensor(flat_params))

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values = self.model(self.states)
        value_loss = (values - self.targets).pow(2).mean()
        value_loss.backward()

        # weight decay
        for param in self.model.parameters():
            value_loss += param.pow(2).sum() * self.l2_reg

        return (
            value_loss.data.cpu().double().numpy(),
            get_flat_grad_from(self.model).data.cpu().double().numpy(),
        )
