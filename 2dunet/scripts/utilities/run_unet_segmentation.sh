#!/usr/bin/env bash


start=`date +%s`
if [ $# -gt 2 ]; then
	model_path=$1
	input_data_path=$2
    output_folder=$3
else
    echo "Not enough arguments -Usage: ./run_unet_segmentation.sh <path/to/model> <path/to/data> <path/to/ouput/dir>"
    exit 1
fi
cd $output_folder
module load unet/dev
unet-setup #This copies settings into output dir
echo -e "\n *** Running 2dunet-prediction. ***\n"
2dunet-predict $model_path $input_data_path
echo "Duration: $((($(date +%s)-$start)/60)) minutes"