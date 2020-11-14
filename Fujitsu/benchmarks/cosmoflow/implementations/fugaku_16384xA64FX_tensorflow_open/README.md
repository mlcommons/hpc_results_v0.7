# 1. Problem

This problem uses TensorFlow Keras implementation for the cosmological parameter prediction benchmark.

# 2. Directions
## Steps to download and verify data
Download the data as follows:

Please download the dataset manually following the instructions from the [CosmoFlow TensorFlow Keras benchmark implementation](https://github.com/sparticlesteve/cosmoflow-benchmark).

## Steps to launch training

### The supercomputer Fugaku
Steps required to launch training on Fugaku:

```
./setup.sh
./init_datasets.sh
./run_and_time.sh
```

# Note
Almost the same implementation is used for closed and open divisions except some minor differences.

Hybrid data-model parallel training is used.
