"""CosmoFlow dataset specification"""

# System imports
import os
import logging
import glob
from functools import partial, reduce
from operator import mul

# External imports
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd

# Local imports
from utils.staging import stage_files

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

class DataPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, tfrecord_paths, idx_paths, num_shards, shard_id, random_shuffle, read_ahead, seed):
        super(DataPipeline, self).__init__(batch_size, num_threads, device_id, seed = seed)
        self.input = ops.TFRecordReader(path = tfrecord_paths,
                                        index_path = idx_paths,
                                        features = {'x' : tfrec.FixedLenFeature((), tfrec.string, ''), 
                                                    'y' : tfrec.FixedLenFeature([4], tfrec.float32, 0.0)},
                                        random_shuffle=random_shuffle,
                                        num_shards=num_shards,
                                        shard_id=shard_id,
                                        read_ahead = read_ahead
                                       )

    def define_graph(self):
        inputs = self.input()
        samples = inputs['x']
        labels = inputs['y']

        samples = samples.gpu()
        labels = labels.gpu()
        return (samples, labels)

def _preproc_data(x, y, shape, apply_log=False, do_augmentation=False):
    """Preproc the data out of dali.
    """

    # Decode the bytes data, convert to float
    x = tf.reshape(x, [-1]+shape+[2])
    x = tf.bitcast(x, tf.int16) # uint8 -> int16
    x = tf.cast(x, tf.float32)

    # Data normalization/scaling
    if apply_log:
        x = tf.math.log(x + tf.constant(1.))
    else:
        # Traditional mean normalization
        x /= (tf.reduce_sum(x) / np.prod(shape))

    # Augmentation [N, x, y, z, C]
    if do_augmentation:
        # random reverse
        rev_idx = tf.random.uniform([3], minval=0, maxval=2, dtype=tf.int32)
        rev_idx = tf.where(rev_idx > 0)
        rev_idx = tf.reshape(rev_idx, [-1])
        rev_idx = tf.add(rev_idx, 1)
        x = tf.reverse(x, rev_idx)

        # random transpose
        random_space_idx = tf.random.shuffle(tf.constant([1,2,3]))
        random_idx = tf.concat([tf.constant([0]), random_space_idx, tf.constant([4])], 0)
        x = tf.transpose(x, perm=random_idx)

    return x, y

def construct_dataset(file_dir, n_samples, batch_size, n_epochs,
                      sample_shape, samples_per_file=1, n_file_sets=1,
                      shard=0, n_shards=1, apply_log=False,
                      randomize_files=False, shuffle=False,
                      shuffle_buffer_size=0, prefetch=4, device_id=0, seed=-1, do_augmentation=False, read_ahead=False):
    """This function takes a folder with files and builds the TF dataset.

    It ensures that the requested sample counts are divisible by files,
    local-disks, worker shards, and mini-batches.
    """

    if n_samples == 0:
        return None, 0

    # Ensure samples divide evenly into files * local-disks * worker-shards * batches
    n_divs = samples_per_file * n_file_sets * n_shards * batch_size
    if (n_samples % n_divs) != 0:
        logging.error('Number of samples (%i) not divisible by %i '
                      'samples_per_file * n_file_sets * n_shards * batch_size',
                      n_samples, n_divs)
        raise Exception('Invalid sample counts')

    # Number of files and steps
    n_files = n_samples // (samples_per_file * n_file_sets)
    n_steps = n_samples // (n_file_sets * n_shards * batch_size)

    # Find the files
    filenames = sorted(glob.glob(os.path.join(file_dir, '*.tfrecord')))
    assert (0 <= n_files) and (n_files <= len(filenames)), (
        'Requested %i files, %i available' % (n_files, len(filenames)))
    if randomize_files:
        np.random.shuffle(filenames)
    filenames = filenames[:n_files]

    # Define the dataset from the list of sharded, shuffled files
    num_threads = 4
    target_shape= [4]

    idx_filenames = ['tfrecord.idx'] * len(filenames) # use same idx_file

    pipeline = DataPipeline(batch_size, num_threads, device_id, filenames, idx_filenames, n_shards, shard, shuffle, read_ahead, seed)
    data = dali_tf.DALIDataset(
        pipeline=pipeline,
        batch_size=batch_size,
        output_shapes=(tuple([batch_size]+[reduce(mul, sample_shape+[2])]), tuple([batch_size]+target_shape)),
        output_dtypes=(tf.uint8, tf.float32),
        device_id=device_id,
    )

    preproc_data = partial(_preproc_data, shape=sample_shape, apply_log=apply_log, do_augmentation=do_augmentation)
    data = data.map(preproc_data)

    return data, n_steps

def get_datasets(data_dir, sample_shape, n_train, n_valid,
                 batch_size, n_epochs, dist, samples_per_file=1,
                 shuffle_train=True, shuffle_valid=False,
                 shard=True, stage_dir=None,
                 prefetch=4, apply_log=False, prestaged=False, seed=-1, do_augmentation=False, validation_batch_size=None, train_staging_dup_factor=1):
    """Prepare TF datasets for training and validation.

    This function will perform optional staging of data chunks to local
    filesystems. It also figures out how to split files according to local
    filesystems (if pre-staging) and worker shards (if sharding).

    Returns: A dict of the two datasets and step counts per epoch.
    """

    data_dir = os.path.expandvars(data_dir)

    # Local data staging
    if stage_dir is not None:
        staged_files = True
        if prestaged:
            if (dist.rank == 0):
                print('data is alreadly staged')
        else:
            if (dist.rank == 0):
                print('data is not staged yet')
            # Stage training data
            stage_files(os.path.join(data_dir, 'train'),
                        os.path.join(stage_dir, 'train'),
                        n_files=n_train, rank=dist.rank, size=dist.size)
            # Stage validation data
            stage_files(os.path.join(data_dir, 'validation'),
                        os.path.join(stage_dir, 'validation'),
                        n_files=n_valid, rank=dist.rank, size=dist.size)

            # Barrier to ensure all workers are done transferring
            if dist.size > 0:
                hvd.allreduce([], name="Barrier")
        data_dir = stage_dir
    else:
        staged_files = False

    # Determine number of staged file sets and worker shards
    if (dist.size // dist.local_size) % train_staging_dup_factor != 0:
        raise Exception('# nodes is not a multiple of train_staging_dup_factor')

    n_train_file_sets = (dist.size // (dist.local_size * train_staging_dup_factor)) if staged_files else 1
    if shard and staged_files:
        n_train_shards = dist.local_size * train_staging_dup_factor
        train_shard = dist.rank % n_train_shards
    elif shard and not staged_files:
        train_shard, n_train_shards = dist.rank, dist.size
    else:
        train_shard, n_train_shards = 0, 1

    n_valid_file_sets = (dist.size // dist.local_size) if staged_files else 1
    if shard and staged_files:
        valid_shard, n_valid_shards = dist.local_rank, dist.local_size
    elif shard and not staged_files:
        valid_shard, n_valid_shards = dist.rank, dist.size
    else:
        valid_shard, n_valid_shards = 0, 1

    device_id = dist.local_rank
    rank_seed = seed + dist.rank if seed >= 0 else -1

    # Construct the training and validation datasets
    train_dataset_args = dict(batch_size=batch_size, n_epochs=n_epochs,
                              sample_shape=sample_shape, samples_per_file=samples_per_file,
                              n_file_sets=n_train_file_sets, shard=train_shard, n_shards=n_train_shards,
                              apply_log=apply_log, prefetch=prefetch)
    train_dataset, n_train_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'train'),
        n_samples=n_train, shuffle=shuffle_train, device_id=device_id, seed=rank_seed, do_augmentation=do_augmentation, **train_dataset_args)

    valid_dataset_args = dict(batch_size=validation_batch_size or batch_size, n_epochs=n_epochs,
                              sample_shape=sample_shape, samples_per_file=samples_per_file,
                              n_file_sets=n_valid_file_sets, shard=valid_shard, n_shards=n_valid_shards,
                              apply_log=apply_log, prefetch=prefetch)
    valid_dataset, n_valid_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'validation'),
        n_samples=n_valid, shuffle=shuffle_valid, device_id=device_id, read_ahead=True, **valid_dataset_args)

    if shard == 0:
        if staged_files:
            logging.info('Using %i locally-staged train file sets and %i locally-staged validation file sets', n_train_file_sets, n_valid_file_sets)
        logging.info('Splitting data into %i train worker shards and validation worker shards', n_train_shards, n_validation_shards)
        n_train_worker = n_train // (samples_per_file * n_train_file_sets * n_train_shards)
        n_valid_worker = n_valid // (samples_per_file * n_valid_file_sets * n_valid_shards)
        logging.info('Each worker reading %i training samples and %i validation samples',
                     n_train_worker, n_valid_worker)

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train_steps=n_train_steps, n_valid_steps=n_valid_steps)
