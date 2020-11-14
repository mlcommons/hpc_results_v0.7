"""
Utilty code for constructing optimizers and scheduling learning rates.
"""

# System
import math
from functools import partial

# Externals
from tensorflow import keras
import horovod.tensorflow.keras as hvd
from mlperf_logging import mllog

def _lr_schedule(epoch, init_lr, peak_lr, n_warmup_epochs, decay_schedule={}):
    """Learning rate schedule function.

    Gives the learning rate as a function of epoch according to
    additional settings:
        base_lr: baseline unscaled learning rate at beginning of training.
        peak_lr: scaled learning at end of warmup period
        n_warmup_epochs: number of linear warmup epochs
        decay_schedule: a dict of epoch number -> decay factor
    """
    # Linear LR warmup
    if epoch < n_warmup_epochs:
        return epoch * (peak_lr - init_lr) / n_warmup_epochs + init_lr
    else:
        decay_name = decay_schedule['name']

        if decay_name == 'step':
            decay_steps = decay_schedule.copy()
            decay_steps.pop('name')
            # Find the most recent decay factor
            decay_factor = 1.
            decay_epoch = 0
            for e, d in decay_steps.items():
                if e >= decay_epoch and e < epoch:
                    decay_epoch, decay_factor = e, d
            return peak_lr * decay_factor
        elif decay_name == 'poly':
            n_decay_epochs = decay_schedule['n_decay_epochs']
            end_lr_factor = decay_schedule['end_factor']
            power = decay_schedule['power']
            decay_epoch = min(epoch - n_warmup_epochs, n_decay_epochs)
            end_lr = peak_lr * end_lr_factor
            return ((peak_lr - end_lr) * (1 - decay_epoch / n_decay_epochs)**power) + end_lr
        else:
            raise Exception('decay name is not specified or not supported')

def get_lr_schedule(base_lr, global_batch_size, base_batch_size=None,
                    scaling=None, n_warmup_epochs=0, warmup_factor=-1, decay_schedule={}, is_root=True):
    """Get the learning rate schedule function"""
    if scaling == 'linear':
        scale_factor = global_batch_size / base_batch_size
    elif scaling == 'sqrt':
        scale_factor = math.sqrt(global_batch_size / base_batch_size)
    else:
        scale_factor = 1.;
    peak_lr = base_lr * scale_factor
    init_lr = peak_lr * warmup_factor if warmup_factor >= 0 else base_lr

    # MLPerf logging
    # NOTE: there is currently a confusing mismatch between the parameter
    # naming convention in this implementation and MLPerf's hyperparameter
    # conventions. Here we define base LR to be the LR at a baseline batch
    # size and the "peak" LR to be the value scaled according to current batch
    # size. We will leave things as-is for now.
    if is_root:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_BASE_LR, value=peak_lr)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_EPOCHS, value=n_warmup_epochs)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_FACTOR, value=warmup_factor if warmup_factor >= 0 else init_lr / peak_lr)

        decay_name = decay_schedule['name']
        if decay_name == 'step':
            decay_steps = decay_schedule.copy()
            decay_steps.pop('name')
            mllogger.event(key=mllog.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                           value=sorted(decay_steps.keys()))
            mllogger.event(key=mllog.constants.OPT_LR_DECAY_FACTOR,
                           value=max(decay_steps.values()) if len(decay_steps)>0 else 1)
    return partial(_lr_schedule, init_lr=init_lr, peak_lr=peak_lr,
                   n_warmup_epochs=n_warmup_epochs,
                   decay_schedule=decay_schedule)

def get_optimizer(name, distributed=False, is_root=True, **opt_args):
    """Configure the optimizer"""

    # MLPerf logging
    if is_root:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_NAME, value=name)

    # Construct the optimizer
    if name == 'LAMB':
        import tensorflow_addons as tfa
        OptType=tfa.optimizers.LAMB
    elif name == 'RAdam':
        import tensorflow_addons as tfa
        OptType=tfa.optimizers.RectifiedAdam
    else:
        OptType = getattr(keras.optimizers, name)
    opt = OptType(**opt_args)

    # Distributed optimizer wrapper
    if distributed:
        opt = hvd.DistributedOptimizer(opt)
        #opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)

    return opt
