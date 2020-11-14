import os
import pandas as pd
import argparse

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



parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()
print_training_summary(args.dir)
#'/home/aca10408mt/CosmoFlow/cosmoflow-benchmark_eval/log/0256/Submit.200822020210.es4.0169e5', True)
