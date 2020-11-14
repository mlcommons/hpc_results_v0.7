#!/bin/bash

MyDir=`readlink -f "$0" | xargs dirname`

. "$ParameterFile"

NumEpoch=`awk '/^ *--num-epochs/ { print $2; }' $ParameterFile | xargs echo`

Echo=true
test "$OMPI_COMM_WORLD_RANK" -eq 0 && Echo=echo
$Echo "number of epochs: $NumEpoch"

if [ ${OMPI_COMM_WORLD_RANK} -eq 0 ]; then
    env > ${LOGDIR}/host_env_0 &
    pip list > ${LOGDIR}/piplist_0 &
fi

CmdProf=""
ProfFile=""
if [ "$UseProf" -gt 0 ]; then
    Remaining=`expr $OMPI_COMM_WORLD_RANK \% $UseProf`
    if [ "$Remaining" -eq 0 ]; then
        ProfFile=`printf '%s/nvprof-rank%04d.prof' "$SGE_LOCALDIR" "$OMPI_COMM_WORLD_RANK"`
        CmdProf="nvprof -o $ProfFile"
    fi
fi

case "$(($OMPI_COMM_WORLD_LOCAL_RANK*2/$OMPI_COMM_WORLD_LOCAL_SIZE))" in
    0) HCA=mlx5_0 ; Node=0 ;;
    1) HCA=mlx5_1 ; Node=1 ;;
esac

export OMPI_MCA_btl_openib_if_include="$HCA"
export NCCL_SOCKET_IFNAME="^bond0,docker0,lo"

### Echo command
$Echo numactl --cpunodebind="$Node" --membind="$Node" ${CmdProf} python "$MyDir/train.py" "${PARAMS[@]}"
### Execute
numactl --cpunodebind="$Node" --membind="$Node" ${CmdProf} python "train.py" "${PARAMS[@]}"

ExitStatus=$?
test -n "$ProfFile" && mv "$ProfFile" "$LOGDIR"
exit $ExitStatus


# End of file

