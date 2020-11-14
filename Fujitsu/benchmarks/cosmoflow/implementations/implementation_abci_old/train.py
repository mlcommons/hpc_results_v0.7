"""
Main training script for the CosmoFlow Keras benchmark
"""

# System imports
import os
import argparse
import logging
import pickle
from types import SimpleNamespace
import re
import random
import subprocess

# External imports
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(logging.ERROR)
import horovod.tensorflow.keras as hvd
from mlperf_logging import mllog
from tensorflow.python.client import timeline

# Local imports
from data import get_datasets
from models import get_model
# Fix for loading Lambda layer checkpoints
from models.layers import *
from utils.optimizers import get_optimizer, get_lr_schedule
from utils.callbacks import TimingCallback, MLPerfLoggingCallback, ProfilingCallback, TerminateOnBaseline
from utils.device import configure_session
from utils.argparse import ReadYaml
from utils.checkpoints import reload_last_checkpoint, reload_checkpoint
from utils.mlperf_logging import configure_mllogger, log_submission_info

# Stupid workaround until absl logging fix, see:
# https://github.com/tensorflow/tensorflow/issues/26691
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

# Keras auto mixed precision
from tensorflow.keras.mixed_precision import experimental as mixed_precision

tf.compat.v1.disable_eager_execution()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/cosmo.yaml')
    add_arg('--output-dir', help='Override output directory')

    # Override data settings
    add_arg('--data-dir', help='Override the path to input files')
    add_arg('--n-train', type=int, help='Override number of training samples')
    add_arg('--n-valid', type=int, help='Override number of validation samples')
    add_arg('--batch-size', type=int, help='Override the batch size')
    add_arg('--n-epochs', type=int, help='Override number of epochs')
    add_arg('--apply-log', type=int, choices=[0, 1], help='Apply log transform to data')
    add_arg('--stage-dir', help='Local directory to stage data to before training')

    # Hyperparameter settings
    add_arg('--conv-size', type=int, help='CNN size parameter')
    add_arg('--fc1-size', type=int, help='Fully-connected size parameter 1')
    add_arg('--fc2-size', type=int, help='Fully-connected size parameter 2')
    add_arg('--hidden-activation', help='Override hidden activation function')
    add_arg('--dropout', type=float, help='Override dropout')
    add_arg('--optimizer', help='Override optimizer type')
    add_arg('--lr', type=float, help='Override learning rate')

    # Other settings
    add_arg('-d', '--distributed', action='store_true')
    add_arg('--rank-gpu', action='store_true',
            help='Use GPU based on local rank')
    add_arg('--resume', type=str,
            help='Resume from last checkpoint')
    add_arg('--print-fom', action='store_true',
            help='Print parsable figure of merit')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--mixed_precision', action='store_true')

    add_arg('--timeline', type=str, help='Output path to json format timeline', default=None)
    add_arg('--prestaged', action='store_true', help='data is already staged to stage-dir')
    add_arg('--seed', type=int, help='Random number seed (not works !)', default=-1)
    add_arg('--target-mae', type=float, help='Stop training when validation mae reachs this')
    add_arg('--do-augmentation', action='store_true')
    add_arg('--validation-batch-size', type=int, help='Batch size for validation')
    add_arg('--train-staging-dup-factor', type=int, help='N times more samples are staged for training')

    return parser.parse_args()

def init_workers(distributed=False):
    if distributed:
        hvd.init()
        return SimpleNamespace(rank=hvd.rank(), size=hvd.size(),
                               local_rank=hvd.local_rank(),
                               local_size=hvd.local_size())
    else:
        return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1)

def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def load_config(args):
    """Reads the YAML config file and returns a config dictionary"""
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Expand paths
    output_dir = config['output_dir'] if args.output_dir is None else args.output_dir
    config['output_dir'] = os.path.expandvars(output_dir)

    # Override data config from command line
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.n_train is not None:
        config['data']['n_train'] = args.n_train
    if args.n_valid is not None:
        config['data']['n_valid'] = args.n_valid
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        config['data']['n_epochs'] = args.n_epochs
    if args.apply_log is not None:
        config['data']['apply_log'] = bool(args.apply_log)
    if args.stage_dir is not None:
        config['data']['stage_dir'] = args.stage_dir
    if args.prestaged:
        config['data']['prestaged'] = args.prestaged
    config['data']['seed'] = args.seed
    if args.do_augmentation:
        config['data']['do_augmentation'] = args.do_augmentation
    if args.validation_batch_size:
        config['data']['validation_batch_size'] = args.validation_batch_size
    config['data']['train_staging_dup_factor'] = args.train_staging_dup_factor or 1

    # Hyperparameters
    if args.conv_size is not None:
        config['model']['conv_size'] = args.conv_size
    if args.fc1_size is not None:
        config['model']['fc1_size'] = args.fc1_size
    if args.fc2_size is not None:
        config['model']['fc2_size'] = args.fc2_size
    if args.hidden_activation is not None:
        config['model']['hidden_activation'] = args.hidden_activation
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.optimizer is not None:
        config['optimizer']['name'] = args.optimizer
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr

    return config

def save_config(config):
    output_dir = config['output_dir']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)

def load_history(output_dir):
    return pd.read_csv(os.path.join(output_dir, 'history.csv'))

def print_training_summary(output_dir, print_fom):
    history = load_history(output_dir)
    if 'val_loss' in history.keys():
        best = history.val_loss.idxmin()
        logging.info('Best result:')
        for key in history.keys():
            logging.info('  %s: %g', key, history[key].loc[best])
        # Figure of merit printing for HPO parsing
        if print_fom:
            print('FoM:', history['val_loss'].loc[best])

def main():
    """Main function"""

    # Initialization
    args = parse_args()

    # Set random seed
    if args.seed != -1:
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = '0'

    dist = init_workers(args.distributed)
    config = load_config(args)
    os.makedirs(config['output_dir'], exist_ok=True)
    config_logging(verbose=args.verbose)

    # Start MLPerf logging
    mllogger = configure_mllogger(config['output_dir'])
    if dist.rank == 0:
        log_submission_info(dist, **config.get('mlperf', {}))

    ### Clear cache
    # if dist.local_rank == 0:
    #     drop_cache_ret = subprocess.run('abci_drop_cache').returncode
    #     if drop_cache_ret != 0:
    #         raise Exception('drop_cache was failed')
    #     mllogger.event(key=mllog.constants.CACHE_CLEAR)
    # if args.distributed: hvd.allreduce([], name="Barrier")

    mllogger.start(key=mllog.constants.INIT_START)

    logging.info('Initialized rank %i size %i local_rank %i local_size %i',
                 dist.rank, dist.size, dist.local_rank, dist.local_size)
    if dist.rank == 0:
        logging.info('Configuration: %s', config)

    # Device and session configuration
    gpu = dist.local_rank if args.rank_gpu else None
    if gpu is not None:
        logging.info('Taking gpu %i', gpu)
    configure_session(gpu=gpu, seed=args.seed, **config.get('device', {}))

    data_config = config['data']
    if dist.rank == 0:
        mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=data_config['n_train'])
        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=data_config['n_valid'])
        mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=data_config['batch_size'] * dist.size)

    # Construct or reload the model
    if dist.rank == 0:
        logging.info('Building the model')
    train_config = config['train']
    initial_epoch = 0
    checkpoint_format = os.path.join(config['output_dir'], 'checkpoint-{epoch:03d}.h5')

#    if args.timeline:
#        run_options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#    else:
#        run_options = tf.compat.v1.RunOptions()
#    run_metadata = tf.compat.v1.RunMetadata()

    if args.resume:
        if os.path.isdir(args.resume):
            resume_checkpoint_format = os.path.join(args.resume, 'checkpoint-{epoch:03d}.h5')
            # Reload model from last checkpoint
            initial_epoch, model = reload_last_checkpoint(
                resume_checkpoint_format, data_config['n_epochs'],
                distributed=args.distributed)
        else:
            basename = os.path.basename(args.resume)
            m = re.fullmatch(r'checkpoint-([0-9][0-9][0-9])\.h5', basename)
            if m is None:
                raise Exception('Can not resume checkpoint file %s' % args.resume)
            initial_epoch = int(m.groups()[0])
            model = reload_checkpoint(args.resume, distributed=args.distributed)

    else:
        if args.mixed_precision:
            # Set auto mixed precision policy
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

        # Build a new model
        model = get_model(**config['model'])
        # Configure the optimizer
        opt = get_optimizer(distributed=args.distributed, is_root = (dist.rank == 0),
                            **config['optimizer'])
        # Compile the model
        model.compile(optimizer=opt, loss=train_config['loss'],
                      metrics=train_config['metrics'],
                      experimental_run_tf_function=False)
#                      options=run_options, run_metadata=run_metadata)

    if dist.rank == 0:
        model.summary()

    # Save configuration to output directory
    if dist.rank == 0:
        config['n_ranks'] = dist.size
        save_config(config)

    # Prepare the callbacks
    if dist.rank == 0:
        logging.info('Preparing callbacks')
    callbacks = []
    if args.distributed:

        # Broadcast initial variable states from rank 0 to all processes.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

        # Average metrics across workers
        callbacks.append(hvd.callbacks.MetricAverageCallback())

    # Learning rate decay schedule
    if 'lr_schedule' in config:
        global_batch_size = data_config['batch_size'] * dist.size
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(
            get_lr_schedule(global_batch_size=global_batch_size, is_root = (dist.rank == 0),
                            **config['lr_schedule'])))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Terminate
    if args.target_mae is not None:
        terminate_callback = TerminateOnBaseline(baseline=args.target_mae)
        callbacks.append(terminate_callback)

#    if dist.rank == 0 and args.timeline:
#        profiling_callback = ProfilingCallback(run_metadata, args.timeline, max(0, datasets['n_train_steps']-16))
#        callbacks.append(profiling_callback)

    # Checkpointing and logging from rank 0 only
    if dist.rank == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))
        callbacks.append(tf.keras.callbacks.CSVLogger(
            os.path.join(config['output_dir'], 'history.csv'), append=args.resume))
        #callbacks.append(tf.keras.callbacks.TensorBoard(
        #    os.path.join(config['output_dir'], 'tensorboard')))
        callbacks.append(MLPerfLoggingCallback())

    # Early stopping
    patience = config.get('early_stopping_patience', None)
    if patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=1e-5, patience=patience, verbose=1))

    if dist.rank == 0:
        logging.debug('Callbacks: %s', callbacks)


    # Run Start
    if args.distributed: hvd.allreduce([], name="Barrier")
    if dist.rank == 0:
        mllogger.end(key=mllog.constants.INIT_STOP)
        mllogger.start(key=mllog.constants.RUN_START)

    # Run staging
    if 'stage_dir' in data_config and 'prestaged' not in data_config:
        if dist.rank == 0:
            mllogger.start(key='staging_start')

        #DataDir=os.environ['DataDir']
        CompressType='xz'
        staging_command='./cpdata_decomp.sh {} {} {} {}'.format(data_config['data_dir'], data_config['stage_dir'], CompressType, data_config['train_staging_dup_factor'])
        staging_ret = subprocess.run(staging_command, shell=True).returncode
        if staging_ret != 0:
            raise Exception('staging was failed')
        data_config['prestaged'] = True

        if args.distributed: hvd.allreduce([], name="Barrier")
        if dist.rank == 0:
            mllogger.start(key='staging_stop')

    # Load the data
    if dist.rank == 0:
        logging.info('Loading data')
    datasets = get_datasets(dist=dist, **data_config)
    logging.debug('Datasets: %s', datasets)

    # Train the model
    if dist.rank == 0:
        logging.info('Beginning training')
    fit_verbose = 1 if (args.verbose and dist.rank==0) else 0

    if config['data']['name'].endswith('dali'):
        with tf.device('/gpu:0'):
            model.fit(datasets['train_dataset'],
                      steps_per_epoch=datasets['n_train_steps'],
                      epochs=data_config['n_epochs'],
                      validation_data=datasets['valid_dataset'],
                      validation_steps=datasets['n_valid_steps'],
                      callbacks=callbacks,
                      initial_epoch=initial_epoch,
                      verbose=fit_verbose)
    else:
        model.fit(datasets['train_dataset'],
                  steps_per_epoch=datasets['n_train_steps'],
                  epochs=data_config['n_epochs'],
                  validation_data=datasets['valid_dataset'],
                  validation_steps=datasets['n_valid_steps'],
                  callbacks=callbacks,
                  initial_epoch=initial_epoch,
                  verbose=fit_verbose)

    # Stop MLPerf timer
    if args.distributed: hvd.allreduce([], name="Barrier")
    if dist.rank == 0:
        mllogger.end(key=mllog.constants.RUN_STOP)

    if dist.rank == 0:
        print('Epoch times : {}'.format(timing_callback.times))

    # Print training summary
    if dist.rank == 0:
        print_training_summary(config['output_dir'], args.print_fom)

    # Finalize
    if dist.rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
