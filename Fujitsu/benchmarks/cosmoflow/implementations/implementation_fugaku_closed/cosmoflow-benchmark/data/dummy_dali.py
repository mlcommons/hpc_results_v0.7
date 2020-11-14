"""
Random dummy dataset specification.
"""

# System
import math

# Externals
import tensorflow as tf


from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

class DummyPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, sample_shape, target_shape):
        super(DummyPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input_sample = ops.Constant(dtype=types.FLOAT,
                                         fdata=0,
                                         shape=sample_shape,
                                         device='cpu')
        self.input_target = ops.Constant(dtype=types.FLOAT,
                                         fdata=0,
                                         shape=target_shape,
                                         device='cpu')

    def define_graph(self):
        samples = self.input_sample()
        labels = self.input_target()

        samples = samples.gpu()
        labels = labels.gpu()
        return (samples, labels)

def construct_dataset(sample_shape, target_shape,
                           batch_size=1, n_samples=32, device_id=0):
    pipeline = DummyPipeline(batch_size, 4, device_id, sample_shape, target_shape)
    data = dali_tf.DALIDataset(
        pipeline=pipeline,
        batch_size=batch_size,
        output_shapes=(tuple([batch_size]+sample_shape), tuple([batch_size]+target_shape)),
        output_dtypes=(tf.float32, tf.float32),
        device_id=device_id
    )
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = False

    data = data.with_options(options)

    return data

def get_datasets(sample_shape, target_shape, batch_size,
                 n_train, n_valid, dist, n_epochs=None, shard=False, seed=-1):
    device_id = dist.local_rank
    train_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size, device_id=device_id)

    valid_dataset = None
    if n_valid > 0:
        valid_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size, device_id=device_id)
    n_train_steps = n_train  // batch_size
    n_valid_steps = n_valid  // batch_size
    if shard:
        n_train_steps = n_train_steps // dist.size
        n_valid_steps = n_valid_steps // dist.size

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
