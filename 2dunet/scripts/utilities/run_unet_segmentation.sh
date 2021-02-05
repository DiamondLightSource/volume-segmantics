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
mkdir -p $output_folder
cd $output_folder
mkdir -p unet-settings; cp /dls_sw/apps/unet/DIAD/unet-segmentation/2dunet/scripts/settings/* unet-settings/.
echo -e "\n *** Running 2dunet-prediction. ***\n"
/dls_sw/apps/unet/conda_env/bin/python /dls_sw/apps/unet/DIAD/unet-segmentation/2dunet/scripts/predict_2d_unet_oo.py $model_path $input_data_path
echo "Duration: $((($(date +%s)-$start)/60)) minutes"