#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 14:38:12 2017

@author: Ashish Katiyar
"""
import torch
from torch.optim.optimizer import Optimizer


class Nadam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)


    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=0.004):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.schedule_decay = schedule_decay
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Nadam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    #
                    state['m_schedule'] = 1.

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Calculate the momentum
                momentum_cache_t = beta1 * (1 - 0.5 * 0.96 ** (state['step'] * self.schedule_decay))
                momentum_cache_t_1 = beta1 * (1 - 0.5 * 0.96 ** ((state['step'] + 1) * self.schedule_decay))

                m_schedule_new = state['m_schedule'] * momentum_cache_t
                m_schedule_next = m_schedule_new * momentum_cache_t_1

                state['m_schedule'] = m_schedule_next

                g_prime = grad / (1. - m_schedule_new)

                # Decay the first and second moment running average coefficient
                # m_t = \beta_1 m_{t-1} + (1 - \beta_1) g
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_prime = exp_avg / (1 - m_schedule_next)

                # v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg_sq_prime = exp_avg_sq / (1. - beta2 ** state['step'])

                exp_avg_bar = (1. - momentum_cache_t) * g_prime + (
                        momentum_cache_t_1 * exp_avg_prime)

                denom = exp_avg_sq_prime.sqrt().add_(group['eps'])

                step_size = group['lr']

                p.data.addcdiv_(-step_size, exp_avg_bar, denom)

        return loss
