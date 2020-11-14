# Ported from torch_optimizer 0.0.1a15: https://pypi.org/project/torch-optimizer/
# Modifited by Fujitsu

import math
import os
import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ('Lamb',)

class Lamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)
        debias: debias adam by (1 - beta**step) (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Lamb(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1904.00962

    Note:
        Reference code: https://github.com/cybertronai/pytorch-lamb
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            num_param = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Normalize gradients by L2 norm of gradient of the entire model
                #torch.nn.utils.clip_grad_norm_(p, 1.0)

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # beta correction used in Apex
                beta1_correction = 1 - pow(beta1, state['step'])
                beta2_correction = 1 - pow(beta2, state['step'])

                exp_avg_unbiased = exp_avg / beta1_correction
                exp_avg_sq_unbiased = exp_avg_sq / beta2_correction

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                #adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                adam_step = exp_avg_unbiased / exp_avg_sq_unbiased.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1
#                print("lr: " + str(group['lr']))
#                print("norm_adam_step: " + str(torch.norm(adam_step)))
#                print("idx: " + str(state['step']))
#                print("number_param: " + str(num_param))
#                print("step_size: " + str(step_size))
#                print("adam_step_size: " + str(adam_step.size()))
#                psize = str(adam_step.size())[11:-1]
                zero_tensor = torch.zeros(adam_step.size(), device=adam_step.device, dtype=adam_step.dtype)
                zero_tensor.add_(adam_step, alpha=-step_size * trust_ratio)
#                print("norm_act_gwm: " + str(torch.norm(zero_tensor).item()))
#                print("Dump, {}, {}, {}, {} ".format(state['step'], num_param, group['lr'], torch.norm(zero_tensor).item()))
                mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK", -1)
                if mpi_rank == "0":
                    with open("param.csv", "a") as f:
                        print("Dump, {}, {}, {}, {} ".format(state['step'], num_param, group['lr'], torch.norm(zero_tensor).item()), file=f)
#                print("trust_ratio: " + str(trust_ratio))
#                print("group_num: " + str(len(group)))
#                print("group_params_num: " + str(len(group['params'])))
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)
                num_param += 1

        return loss
