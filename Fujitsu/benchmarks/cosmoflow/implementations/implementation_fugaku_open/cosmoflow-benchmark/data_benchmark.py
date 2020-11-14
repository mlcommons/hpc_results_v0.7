"""Data loading benchmark code for cosmoflow-benchmark

This script can be used to test just the data-loading part of the CosmoFlow
application to understand I/O performance.
"""

# System imports
import argparse
import time
import pprint
from types import SimpleNamespace

# External imports
import tensorflow as tf
#import horovod.tensorflow.keras as hvd

# Local imports
from data import get_datasets


import os
import re
import psutil

def peak_memory():
    pid = os.getpid()
    with open(f'/proc/{pid}/status') as f:
        for line in f:
            if not line.startswith('VmHWM:'):
                continue
            return int(re.search('[0-9]+', line)[0])
    raise ValueError('Not Found')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data-dir', default='/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf')
    add_arg('--n-samples', type=int, default=512)
    add_arg('--batch-size', type=int, default=4)
    add_arg('--n-epochs', type=int, default=1)
    add_arg('--inter-threads', type=int, default=2)
    add_arg('--intra-threads', type=int, default=32)
    add_arg('-d', '--distributed', action='store_true')
    add_arg('--use-cache', action='store_true')
    add_arg('--cache-as-fp32', action='store_true')
    return parser.parse_args()

def init_workers(distributed=False):
    if distributed:
        hvd.init()
        return SimpleNamespace(rank=hvd.rank(), size=hvd.size(),
                               local_rank=hvd.local_rank(),
                               local_size=hvd.local_size())
    else:
        return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, data_parallel_size=1, data_parallel_rank=0, data_parallel_local_size=1, data_parallel_local_rank=0, model_parallel_size=(1,1), model_parallel_rank=(0,0))

def main():
    # Parse command line arguments
    args = parse_args()

    # Session setup
    tf.compat.v1.enable_eager_execution(
        config=tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=args.inter_threads,
            intra_op_parallelism_threads=args.intra_threads))

    dist = init_workers(args.distributed)

    used_mem = psutil.Process().memory_full_info().rss /1000/1000
    print('first memory {}'.format(used_mem))

    # Load the dataset
    data = get_datasets(name='cosmo',
                        data_dir=args.data_dir,
                        sample_shape=[128, 128, 128, 4],
                        n_train=args.n_samples,
                        n_valid=0,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        apply_log=True,
                        shard=True,
                        do_augmentation=True,
                        dist=dist,
                        use_cache=args.use_cache,
                        cache_as_fp32=args.cache_as_fp32)

    print('after datacreateion, memory {}'.format(used_mem))

    if dist.rank == 0:
        pprint.pprint(data)

    if args.distributed: hvd.allreduce([], name="Barrier")
    n_train_steps= data['n_train_steps']
    start_time = time.perf_counter()
    epoch_start_time = start_time
    for idx, (x, y) in enumerate(data['train_dataset']()):
        if idx % n_train_steps == 0:
            print(x[0,0,0,0,:])
        if idx % n_train_steps == 1:
            print(x[0,0,0,0,:])
        if idx % n_train_steps == 2:
            print(x[0,0,0,0,:])
        if (idx+1)%n_train_steps == 0:
            cur = time.perf_counter()
            d = cur - epoch_start_time
            epoch_start_time= cur
            used_mem = psutil.Process().memory_full_info().rss /1000/1000
            #used_mem = peak_memory() /1000
            print('Epoch {} {}img/s {}MB'.format((idx+1)//n_train_steps, args.n_samples / d, used_mem))

        pass
        # Perform a simple operation
        # tf.math.reduce_sum(x)
        # tf.math.reduce_sum(y)
    if args.distributed: hvd.allreduce([], name="Barrier")
    duration = time.perf_counter() - start_time

    if dist.rank == 0:
        print('Total time: %.4f s' % duration)
        print('Throughput: %.4f samples/s' % (args.n_samples * args.n_epochs / duration))

        print('All done!')

if __name__ == '__main__':
    main()
