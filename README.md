# Volume Segmantics

A toolkit for semantic segmentation of volumetric data using PyTorch deep learning models.

Given a 3d image volume and corresponding dense labels (the segmentation), a 2d model is trained on image slices taken along the x, y, and z axes. The method is optimised for small training datasets, e.g a single $384^3$ pixel dataset. To achieve this, all models use pretrained encoders and image augmentations are used to expand the size of the training dataset.

This work utilises the abilities afforded by the excellent [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) library. Also the metrics and loss functions used make use of the hard work done by Adrian Wolny in his [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) repository. 

## Installation

At present, the easiest way to install is to create a new conda enviroment or virtualenv with python (ideally >= version 3.8) and pip, activate the envionment and `pip install volume-segmantics`.

## Configuration and command line use

After installation, two new commands will be available from your terminal whilst your environment is activated, `2d-model-train` and `2d-model-predict`.

These commands require access to some settings stored in YAML files. These need to be located in a directory named `volseg-settings` within the directory where you are running the commands. The settings files can be copied from [here](https://gitlab.diamond.ac.uk/data-analysis/imaging/unet-segmentation/-/tree/packaging/settings). 

The file `2d_model_train_settings.yaml` can be edited in order to change training parameters such as number of epochs, loss functions, evaluation metrics and also model and encoder architectures. The file `2d_model_predict_settings.yaml` can be edited to change parameters such as the prediction "quality" e.g "low" quality refers to prediction of the volume segmentation by taking images along a single axis (images in the (x,y) plane). For "medium" and "high" quality, predictions are done along 3 axes and in 12 directions respectively, before being combined by maximum probability. 

### For training a 2d model on a 3d image volume and corresponding labels
Run the following command. Input files can be in HDF5 or multipage TIFF format.

```shell
2d-model-train --data path/to/image/data.h5 --labels path/to/corresponding/segmentation/labels.h5
```

A model will be trained according to the settings defined in `/volseg-settings/2d_model_train_settings.yaml` and saved to your working directory. In addition, a figure showing "ground truth" segmentation vs model segmentation for some images in the validation set will be saved. 

##### For 3d volume segmentation prediction using a 2d model
Run the following command. Input image files can be in HDF5 or multipage TIFF format.

```shell
2d-model-predict path/to/model_file.pytorch path/to/data_for_prediction.h5
```

The input data will be segmented using the input model following the settings specified in `volseg-settings/2d_model_predict_settings.yaml`. An HDF5 file containing the segmented volume will be saved to your working directory.
