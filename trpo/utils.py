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
def get_ret_and_gae(rewards, values, dones, gamma, lambd):
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
    x = torch.zeros(b.size())
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
    print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(),
              expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
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
def trpo_step(model, states, actions, advantages, max_kl, damping):
    """Calculate TRPO loss."""
    means, log_stds, stds = model(states)
    fixed_log_prob = normal_log_density(actions, means,
                                        log_stds, stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                means, log_stds, stds = model(states)
        else:
            means, log_stds, stds = model(states)

        log_prob = normal_log_density(actions, means, log_stds, stds)
        action_loss = -advantages * torch.exp(log_prob - fixed_log_prob)
        return action_loss.mean()

    def get_kl():
        # TODO: check this calculation is correct.
        mean1, log_std1, std1 = model(states)

        mean0 = torch.Tensor(mean1.data)
        log_std0 = torch.Tensor(log_std1.data)
        std0 = torch.Tensor(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / \
            (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                       for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss


# taken from https://github.com/ikostrikov/pytorch-trpo
# TODO: this function should have only 1 arg.
def get_value_loss(flat_params, model, states, targets):
    """Return value loss."""
    set_flat_params_to(model, torch.Tensor(flat_params))

    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.fill_(0)

    values = model(states)
    value_loss = (values - targets).pow(2).mean()
    value_loss.backward()

    return (value_loss.data.double().numpy(),
            get_flat_grad_from(model).data.double().numpy())
