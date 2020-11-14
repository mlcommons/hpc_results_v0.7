#!/bin/bash
#SBATCH -N 16
#SBATCH --ntasks-per-node 4
##SBATCH --gpus-per-task 1
#SBATCH -t 15:00:00
#SBATCH -J cosmoflow
#SBATCH -p rtx
#SBATCH -o logs/%x-%j.out
#SBATCH -A allocation-id

isImpi=1    # Intel MPI or MVAPICH2
isDryRun=0  # Run a small case
ppn=4
ranks=$(( $SLURM_NNODES * $ppn ))

echo "SLURM_NNODES =$SLURM_NNODES ppn=$ppn"
module purge

source /scratch3/05231/aruhela/libs/anaconda/bin/activate 
if [[ "$isImpi" == "1" ]]
then
    conda activate mlperf_cosmo_scratch3
    conda env list    
    ml gcc/9.1.0 cuda/10.0 cudnn/7.6.2 nccl/2.4.7 cmake impi
    set -x
    #export I_MPI_HYDRA_DEBUG=1
    #export I_MPI_DEBUG="3"
    set +x
else
    module purge
    conda activate mlperf_mv2
    conda env list
    module load gcc/8.3.0 cuda/10.0 cudnn/7.6.2 cmake mvapich2-gdr
    set -x
    #export MV2_DEBUG_SHOW_BACKTRACE=1
    export MV2_SHOW_ENV_INFO=1
    export MV2_SHOW_CPU_BINDING=1
    export MV2_PATH=$MPICH_HOME
    export MV2_SUPPORT_DL=1
    export MV2_USE_CUDA=1
    export MV2_USE_GDRCOPY=0
    export LD_PRELOAD=/scratch1/05231/aruhela/libs/tcmalloc/usr/lib64/libtcmalloc_minimal.so.4
    #unset LD_PRELOAD
    set +x
fi

module list

set -x
export CUDA_HOME=$TACC_CUDA_DIR
export CUDNN_ROOT=$TACC_CUDNN_DIR
export CUDNN_INCLUDE_DIR=$TACC_CUDNN_INC
export CUDNN_LIBRARY=$TACC_CUDNN_LIB
set +x

rm core*

############ Print environment #############
module list
myprintenv.sh
which mpicc
echo -e "\n --- MVAPICH2 Settings -- "
env | grep MV2
echo -e "\n --- Intel Settings -- "
env | grep -i IMPI
echo -e "\n --- CUD Settings -- "
env | grep CUD
echo -e "\n ----- \n"
#export HOROVOD_TIMELINE=./timeline.json

conda env list
conda list


echo "Starting train_cgpu.sh"
SECONDS=0
set -x

mkdir logs resultsdir dryresultsdir

mytrain="--n-train 262144"
myvalid="--n-valid 65536"
mybatchsize="--batch-size 1"
myoutdir="--output-dir resultsdir/result-N$SLURM_NNODES-n$SLURM_NTASKS-g$SLURM_GPUS_PER_TASK-j$SLURM_JOBID"

if [[ "$isDryRun" == "1"  ]]
then
   echo "Running DryRun" 
   myconfig="configs/cosmo_dryrun.yaml"
   mytrain="--n-train 32"
   myvalid="--n-valid 32"
   myepochs="--n-epochs 30"
   mybatchsize="--batch-size 1"
   myoutdir="--output-dir dryresultsdir/result-N$SLURM_NNODES-n$SLURM_NTASKS-g$SLURM_GPUS_PER_TASK-j$SLURM_JOBID"
fi

echo "started at "
date

if [[ "$isImpi" == "1" ]]
then
   ibrun -n $ranks python3 train.py -d $myconfig --rank-gpu $mytrain $myvalid $mybatchsize $myepochs $myoutdir $@ 
else
   gen_hostfile.sh $ppn # generate a hostfile   
   mpirun_rsh -export-all -np $ranks -hostfile hosts python3 train.py -d --rank-gpu $mytrain $myvalid $mybatchsize $myepochs $myoutdir $@
fi

echo "finished at "
date
duration=$SECONDS
echo "Run finished for in $duration seconds"
echo "`date` = $(($duration / 3600)) hours : $(($duration / 60)) minutes : $(($duration % 60)) seconds"

