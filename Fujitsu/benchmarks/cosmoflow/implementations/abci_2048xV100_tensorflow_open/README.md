# 1. Problem

This problem uses TensorFlow Keras implementation for the cosmological parameter prediction benchmark.

# 2. Directions
## Steps to download and verify data
Download the data as follows:

Please download the dataset manually following the instructions from the [CosmoFlow TensorFlow Keras benchmark implementation](https://github.com/sparticlesteve/cosmoflow-benchmark).

## Steps to launch training

### ABCI (FUJITSU PRIMERGY CX2570 M4)
Please edit `DataDir` and `StageDir` variables in `run_and_time.sh`.
Steps required to launch training on ABCI (FUJITSU PRIMERGY CX2570 M4):

```
./setup.sh
./init_datasets.sh
./run_and_time.sh
```

# Note
This implementation is compatible with the hpc 0.5.0 compliance checker as of August 27th, 2020.
