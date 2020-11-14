import os
import pandas as pd
import argparse
import pickle

def load_history(output_dir):
    return pd.read_csv(os.path.join(output_dir, 'history.csv'))

def print_training_summary(output_dir):
    history = load_history(output_dir)
    if 'val_mae' in history.keys():
        best = history.val_mae.idxmin()

        best_val_mae = history['val_mae'].loc[best]
        best_epoch = 1 + history['epoch'].loc[best]
        mean_time = history[1:].time.mean()

        reached_idxes = history.query('val_mae < 0.124').index
        if reached_idxes.empty:
            first_reached_epoch = '-'
        else:
            first_reached_epoch = 1 + history['epoch'].loc[reached_idxes[0]]

        print(mean_time, first_reached_epoch, best_val_mae, best_epoch)

def print_params(output_dir):
    config_file = os.path.join(output_dir, 'config.pkl')
    l = []

    with open(config_file, 'rb') as f:
        config = pickle.load(f)
        #print(config)
        data = config['data']
        opt = config['optimizer']
        lr_sched = config['lr_schedule']
        lr_decay_sched = lr_sched['decay_schedule']
        model = config['model']
        train = config['train']

        # githash
        l.append('?')
        # Nodes
        l.append('?')

        # Batchsize
        l.append(data['batch_size'])

        # n-epoch
        l.append(data['n_epochs'])

        # mixed-fp16
        l.append('?')

        # dup data
        l.append(data['train_staging_dup_factor'])

        # seed
        l.append(data['seed'])

        # Optimizer
        #   name
        l.append(opt['name'])
        #   momentum
        l.append(opt['momentum'])
        #   decay
        l.append(opt.get('decay', '-'))
        #   weight decay
        l.append(opt.get('weight_decay', '-'))
        #   epsilon
        l.append(opt.get('epsilon', '-'))

        # Learning Rate
        #   base_lr
        l.append(lr_sched['base_lr'])
        #   scaling
        l.append(lr_sched['scaling'])
        #   base_bs
        l.append(lr_sched['base_batch_size'])

        # Warmup
        #   n_warmup_epochs
        l.append(lr_sched['n_warmup_epochs'])
        #   warmup_factor
        l.append(lr_sched['warmup_factor'])

        # Step decay
        lr_decay_name = lr_decay_sched['name']
        if lr_decay_name == 'step':
            decay_steps = lr_decay_sched.copy()
            decay_steps.pop('name')
            for e in sorted(decay_steps.keys()):
                l.append(e)
                l.append(decay_steps[e])
        else:
            l.extend(['-'] * 4)
        if lr_decay_name == 'poly':
            l.append(lr_decay_sched['n_decay_epochs'])
            l.append(lr_decay_sched['end_factor'])
            l.append(lr_decay_sched['power'])
        else:
            l.extend(['-'] * 3)
        if lr_decay_name not in ('step', 'poly'):
            if lr_decay_name == 'htd':
                l.append('"{}({},{},{},{})"'.format(lr_decay_name, lr_decay_sched['n_decay_epochs'], lr_decay_sched['end_factor'], lr_decay_sched['L'], lr_decay_sched['U']))
            elif lr_decay_name == 'cos':
                l.append('"{}({},{})"'.format(lr_decay_name, lr_decay_sched['n_decay_epochs'], lr_decay_sched['end_factor']))
            else:
                l.append('?')
        else:
            l.append('-')

        # do_augmentation
        l.append(data['do_augmentation'])

        # dropout
        l.append(model['dropout'])

        # loss_func
        l.append(train['loss'])

        # kernel size
        l.append(model.get('kernel_size', 2))
    print(*l, end=' ')


def read_parameters(output_dir):
    parameters_file = os.path.join(output_dir, 'parameters')
    l = []

    with open(parameters_file, 'r') as f:
        while True:
            line = f.readline()
            if line:
                print(line)
            else:
                break

parser = argparse.ArgumentParser()
parser.add_argument('dirs', metavar='D', type=str, nargs='+')
args = parser.parse_args()
#'/home/aca10408mt/CosmoFlow/cosmoflow-benchmark_eval/log/0256/Submit.200822020210.es4.0169e5', True)

#read_parameters(args.dir)

for d in args.dirs:
    print(d, end=' ')
    print_params(d)
    print('? ? ?', end=' ')
    print_training_summary(d)
