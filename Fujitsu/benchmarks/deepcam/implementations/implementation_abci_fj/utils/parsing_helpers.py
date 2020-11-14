# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import re
import numpy as np
import torch
import torch.optim as optim

#polynomial scheduler
have_polynomial_scheduler = False
try:
    from utils.torch_poly_lr_decay import PolynomialLRDecay
    have_polynomial_scheduler = True
except ImportError:
    pass


def get_lr_schedule(start_lr, scheduler_arg, optimizer, last_step = -1):
    #add the initial_lr to the optimizer
    optimizer.param_groups[0]["initial_lr"] = start_lr

    #now check
    if scheduler_arg["type"] == "multistep":
        milestones = [ int(x) for x in scheduler_arg["milestones"].split() ]
        gamma = float(scheduler_arg["decay_rate"])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch = last_step)
    elif scheduler_arg["type"] == "polynomial":
        if have_polynomial_scheduler:
            max_decay_steps = int(scheduler_arg["max_decay_steps"])
            end_learning_rate = float(scheduler_arg["end_learning_rate"])
            power = float(scheduler_arg["power"])
            return PolynomialLRDecay(optimizer, max_decay_steps=max_decay_steps, end_learning_rate=end_learning_rate, power=power)
        else:
            raise ImportError("Error, {} is not installed.".format("torch_poly_lr_decay"))
    else:
        raise ValueError("Error, scheduler type {} not supported.".format(scheduler_arg["type"]))


def get_lr(scheduler, scheduler_arg):
    return scheduler.get_last_lr()[0]
