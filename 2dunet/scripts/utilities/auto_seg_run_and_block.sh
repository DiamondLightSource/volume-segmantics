#!/usr/bin/env bash

script_path=/dls_sw/apps/unet/DIAD/unet-segmentation/2dunet/scripts/utilities/run_unet_segmentation.sh
# Script to submit unet segmentation job to cluster and block until complete
if [ $# -gt 2 ]; then
	model_path=$1
	input_data_path=$2
    output_folder=$3
else
    echo "Not enough arguments -Usage: ./auto_seg_run_and_block.sh <path/to/model> <path/to/data> <path/to/ouput/dir>"
    exit 1
fi
module load hamilton
JOBID=$(qsub -P k11 -q all.q -N unet_segment -j yes -pe smp 2 -l gpu=1 -l gpu_arch=Volta -o ${output_folder} -e ${output_folder} ${script_path} ${model_path} ${input_data_path} ${output_folder})
JOBID=`echo $JOBID | awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}'`
while qstat -j $JOBID &> /dev/null; do
    sleep 5;
done;
