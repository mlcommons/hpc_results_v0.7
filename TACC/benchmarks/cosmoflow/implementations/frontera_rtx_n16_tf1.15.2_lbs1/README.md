# Application : CosmoFlow 

## Download 
Repository
  - Clone the git repository for Cosmoflow from [URL](https://github.com/sparticlesteve/cosmoflow-benchmark.git) with following syntax.

  - Syntax : git clone https://github.com/sparticlesteve/cosmoflow-benchmark.git

Dataset

  - Download the pre-processed dataset (cosmoUniverse_2019_05_4parE_tf.tgz) in TFRecord format  from [URL](https://portal.nersc.gov/project/dasrepo/cosmoflow-benchmark/)

  - [URL](https://github.com/sparticlesteve/cosmoflow-benchmark/blob/master/README.md) provides complete instructions to download dataset. Globus is preferred way to download the 1T size dataset to save time and avoind network connection issues.

  - For getting started, there is also a small tarball (179MB) with 32 training samples and 32 validation samples, called cosmoUniverse_2019_05_4parE_tf_small.tgz.

## Setup
Follow the steps given in setup.sh for Frontera RTX Queue specific instructions.

## Run
1. Copy run_time.sh in tf_cosmoflow/scripts directory.
2. cd tf_cosmoflow    # Cosmoflow top directory
3. sbatch -N Number-of-nodes scripts/run_time.sh

Note : The script "run_time.sh" sets local_batch_size=1 which can be configured in tf_cosmoflow/configs/cosmo.yaml or overrided in run_time.sh
