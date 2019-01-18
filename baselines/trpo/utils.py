# -*- coding: utf-8 -*-
"""Utility functions repeatedly used for TRPO.

This module has TRPO util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://arxiv.org/abs/1502.05477
"""

import math
import numpy as np
import torch
from torch.distributions.kl import kl_divergence


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def decompose_memory(memory):
    """Decompose states, actions, rewards, dones from the memory."""
    memory = np.array(memory)
    states = torch.from_numpy(np.vstack(memory[:, 0])).float().to(device)
    actions = torch.from_numpy(np.vstack(memory[:, 1])).float().to(device)
    rewards = torch.from_numpy(np.vstack(memory[:, 2])).float().to(device)
    dones = torch.from_numpy(
                np.vstack(memory[:, 3]).astype(np.uint8)).float().to(device)

    return states, actions, rewards, dones


# taken from https://github.com/ikostrikov/pytorch-trpo
def get_flat_params_from(model):
    """Return flat_params from the model parameters."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


# taken from https://github.com/ikostrikov/pytorch-trpo
def set_flat_params_to(model, flat_params):
    """Set the model parameters as flat_params."""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


# taken from https://github.com/ikostrikov/pytorch-trpo
def get_flat_grad_from(net, grad_grad=False):
    """Return flat grads of the model parameters."""
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


# taken from https://github.com/ikostrikov/pytorch-trpo
def get_gae(rewards, values, dones, gamma, lambd):
    """Calculate returns and GAEs."""
    masks = 1 - dones
    returns = torch.zeros_like(rewards)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * lambd * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    return returns, advantages


# taken from https://github.com/ikostrikov/pytorch-trpo
def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
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


# taken from https://github.com/ikostrikov/pytorch-trpo
def linesearch(model, f, x, fullstep, expected_improve_rate,
               max_backtracks=10, accept_ratio=.1):
    """Backtracking linesearch.

    where expected_improve_rate is the slope the slope dy/dx
    at the initial point.
    """
    fval = f(True).data
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + torch.tensor(stepfrac).to(device) * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, x


# taken from https://github.com/ikostrikov/pytorch-trpo
def normal_log_density(x, mean, log_std, std):
    """Calculate log density of nomal distribution."""
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / \
        (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


# taken from https://github.com/ikostrikov/pytorch-trpo
def trpo_step(old_actor, actor, states, actions,
              advantages, max_kl, damping):
    """Calculate TRPO loss."""
    _, old_dist = old_actor(states)
    old_log_prob = old_dist.log_prob(actions)

    def get_loss(volatile=False):
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

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, actor.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                       for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]
    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    prev_params = get_flat_params_from(actor)

    success, new_params = linesearch(actor, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])

    set_flat_params_to(actor, new_params)

    return loss


# taken from https://github.com/ikostrikov/pytorch-trpo
class ValueLoss(object):
    """Value Loss calculator.

    Attributes:
        model (nn.Module): model to predict values
        states (torch.Tensor): current states
        targets (torch.Tensor): current returns
        l2_reg (float): weight decay rate

    """

    def __init__(self, model, states, targets, l2_reg):
        """Initialization."""
        self.model = model
        self.states = states
        self.targets = targets
        self.l2_reg = l2_reg

    def __call__(self, flat_params):
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

        return (value_loss.data.to('cpu').double().numpy(),
                get_flat_grad_from(self.model).data.to('cpu').double().numpy())
