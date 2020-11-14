import glob
import re
import os
import pandas as pd

# Log files
resultlist = glob.glob('log/**/stdout.txt',recursive=True)

# Generate result file
resultfilename = 'throuput_check.csv'
result_file = open(resultfilename, 'w')
columinfo =  'train speed per gpu (data/sec), min step time, time unit, local batchsize, # of procs, node info'
result_file.write(columinfo)

# Check each result file
for filename in resultlist:
    with open(filename) as f:
        lines = f.readlines()

    # Get 's/step' info
    lines_strip = [line.strip() for line in lines]
    time_step = [line for line in lines_strip if 's/step' in line]
    time_step_idx = [ line.find('s/step') for line in time_step ]
    time_step = [ line[time_step_idx[i]-4:time_step_idx[i]+6] for i, line in enumerate(time_step) ]
    time_step_float = [ float(re.sub('[^0-9]','', line)) for line in time_step ]

    time_unit = 'ms' if len([ line.find('ms/step') for line in time_step ]) else 's'
    
    if len(time_step_float):
        # Get min step time
        min_time = min(time_step_float)

        # Get local batchsize        
        paramfile = os.path.dirname(filename) + '/parameters'
        with open(paramfile) as f:
            lines = f.readlines()
        batchsize = [ int(re.sub('[^0-9]','',line)) for line in lines if ('batch-size' in line) and (not '#' in line) ]
        # Training speed (data/sec) per GPU
        train_speed_1gpu = batchsize[0] / min_time 
        train_speed_1gpu = train_speed_1gpu * 1000 if time_unit is 'ms' else train_speed_1gpu

        # Get node info
        hostfile = os.path.dirname(filename) + '/hostfile'
        with open(hostfile) as f:
            lines = f.readlines()
        nodeinfo = [ line[:5] for line in lines ]

        # Create summary 
        resultline = '\n' + str(train_speed_1gpu) + ',' + str(min_time) + ',' + time_unit \
                        + ',' + str(batchsize[0]) + ',' + str(len(nodeinfo)) + ',' + ' '.join(nodeinfo)

        result_file.write(resultline)

result_file.close()


# Sort by train speed per gpu
df = pd.read_csv(resultfilename, sep=',')
df = df.sort_values(by='train speed per gpu (data/sec)')
df.to_csv('sorted-'+ resultfilename)
