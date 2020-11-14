# 1. Problem

This problem uses PyTorch implementation for the climate segmentation benchmark.

# 2. Directions
## Steps to download and verify data
Download the data as follows:

Please download the dataset manually following the instructions from the [Deep Learning Climate Segmentation Benchmark website](https://github.com/azrael417/mlperf-deepcam). No preprocessing was performed on the raw hdf5 data.

## Steps to launch training

### ABCI (FUJITSU PRIMERGY CX2570 M4)
Steps required to launch 256 nodes and four GPUs per node training on ABCI (FUJITSU PRIMERGY CX2570 M4):

```
./setup.sh
./init_datasets.sh
./run_and_time.sh 256
```
