# Volume Segmantics

A toolkit for semantic segmentation of volumetric data using PyTorch deep learning models.

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04691/status.svg)](https://doi.org/10.21105/joss.04691) ![example workflow](https://github.com/DiamondLightSource/volume-segmantics/actions/workflows/tests.yml/badge.svg) ![example workflow](https://github.com/DiamondLightSource/volume-segmantics/actions/workflows/release.yml/badge.svg)

Volume Segmantics provides a simple command-line interface and API that allows researchers to quickly train a variety of 2D PyTorch segmentation models (e.g.  U-Net, U-Net++, FPN, DeepLabV3+) on their 3D datasets. These models use pre-trained encoders, enabling fast training on small datasets. Subsequently, the library enables using these trained models to segment larger 3D datasets, automatically merging predictions made in orthogonal planes and rotations to reduce artifacts that may result from predicting 3D segmentation using a 2D network.  

Given a 3d image volume and corresponding dense labels (the segmentation), a 2d model is trained on image slices taken along the x, y, and z axes. The method is optimised for small training datasets, e.g a single dataset in between $128^3$ and $512^3$ pixels. To achieve this, all models use pre-trained encoders and image augmentations are used to expand the size of the training dataset.

This work utilises the abilities afforded by the excellent [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) library in combination with augmentations made available via [Albumentations](https://albumentations.ai/). Also the metrics and loss functions used make use of the hard work done by Adrian Wolny in his [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) repository. 

## Requirements

A machine capable of running CUDA enabled PyTorch version 1.7.1 or greater is required. This generally means a reasonably modern NVIDIA GPU. The exact requirements differ according to operating system. For example on Windows you will need Visual Studio Build Tools as well as CUDA Toolkit installed see [the CUDA docs](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for more details. 

## Installation

The easiest way to install the package is to first create a new conda environment or virtualenv with python (ideally >= version 3.8) and also pip, then activate the environment and `pip install volume-segmantics`. If a CUDA-enabled build of PyTorch is not being installed by pip, you can try `pip install volume-segmantics --extra-index-url https://download.pytorch.org/whl` this particularity seems to be an issue on Windows. 

## Configuration and command line use

After installation, two new commands will be available from your terminal whilst your environment is activated, `model-train-2d` and `model-predict-2d`.

These commands require access to some settings stored in YAML files. These need to be located in a directory named `volseg-settings` within the directory where you are running the commands. The settings files can be copied from [here](https://github.com/DiamondLightSource/volume-segmantics/releases/download/v0.3.1/volseg-settings.zip). 

The file `2d_model_train_settings.yaml` can be edited in order to change training parameters such as number of epochs, loss functions, evaluation metrics and also model and encoder architectures. The file `2d_model_predict_settings.yaml` can be edited to change parameters such as the prediction "quality" e.g "low" quality refers to prediction of the volume segmentation by taking images along a single axis (images in the (x,y) plane). For "medium" and "high" quality, predictions are done along 3 axes and in 12 directions (3 axes, 4 rotations) respectively, before being combined by maximum probability. 

### For training a 2d model on a 3d image volume and corresponding labels
Run the following command. Input files can be in HDF5 or multi-page TIFF format.

```shell
model-train-2d --data path/to/image/data.h5 --labels path/to/corresponding/segmentation/labels.h5
```

Paths to multiple data and label volumes can be added after the `--data` and `--labels` flags respectively. A model will be trained according to the settings defined in `/volseg-settings/2d_model_train_settings.yaml` and saved to your working directory. In addition, a figure showing "ground truth" segmentation vs model segmentation for some images in the validation set will be saved. 

### For 3d volume segmentation prediction using a 2d model
Run the following command. Input image files can be in HDF5 or multi-page TIFF format.

```shell
model-predict-2d path/to/model_file.pytorch path/to/data_for_prediction.h5
```

The input data will be segmented using the input model following the settings specified in `volseg-settings/2d_model_predict_settings.yaml`. An HDF5 file containing the segmented volume will be saved to your working directory.

### Tutorial using example data

A tutorial is available [here](training_data/README.md) that provides a walk-through of how to segment blood vessels from synchrotron X-ray micro-CT data collected on a sample of human placental tissue.

## Currently supported model architectures and encoders

The model architectures which are currently available and tested are: 
- U-Net
- U-Net++
- FPN
- DeepLabV3
- DeepLabV3+
- MA-Net
- LinkNet
- PAN

The pre-trained encoders that can be used with these architectures are: 
- ResNet-34
- ResNet50
- ResNeXt-50_32x4d
- Efficientnet-b3
- Efficientnet-b4
- Resnest50d\*
- Resnest101e\*

\* Encoders with asterisk not compatible with PAN.

## Using the API

You can use the functionality of the package in your own program via the API, this is [documented here](https://diamondlightsource.github.io/volume-segmantics/). This interface is the one used by [SuRVoS2](https://github.com/DiamondLightSource/SuRVoS2), a client/server GUI application that allows fast annotation and segmentation of volumetric data. 

## Contributing

We welcome contributions from the community. Please take a look at out [contribution guidelines](https://github.com/DiamondLightSource/volume-segmantics/blob/main/CONTRIBUTING.md) for more information.

## Citation

If you use this package for you research, please cite:

[King O.N.F, Bellos, D. and Basham, M. (2022). Volume Segmantics: A Python Package for Semantic Segmentation of Volumetric Data Using Pre-trained PyTorch Deep Learning Models. Journal of Open Source Software, 7(78), 4691. doi: 10.21105/joss.04691](https://doi.org/10.21105/joss.04691)

```bibtex
@article{King2022,
    doi = {10.21105/joss.04691},
    url = {https://doi.org/10.21105/joss.04691},
    year = {2022},
    publisher = {The Open Journal},
    volume = {7},
    number = {78},
    pages = {4691},
    author = {Oliver N. F. King and Dimitrios Bellos and Mark Basham},
    title = {Volume Segmantics: A Python Package for Semantic Segmentation of Volumetric Data Using Pre-trained PyTorch Deep Learning Models},
    journal = {Journal of Open Source Software} }
```

## References

**Albumentations**

Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M., and Kalinin, A.A. (2020). Albumentations: Fast and Flexible Image Augmentations. Information 11. [https://doi.org/10.3390/info11020125](https://doi.org/10.3390/info11020125)

**Segmentation Models PyTorch**

Yakubovskiy, P. (2020). Segmentation Models Pytorch. [GitHub](https://github.com/qubvel/segmentation_models.pytorch)

**PyTorch-3dUnet**

Wolny, A., Cerrone, L., Vijayan, A., Tofanelli, R., Barro, A.V., Louveaux, M., Wenzl, C., Strauss, S., Wilson-Sánchez, D., Lymbouridou, R., et al. (2020). Accurate and versatile 3D segmentation of plant tissues at cellular resolution. ELife 9, e57613. [https://doi.org/10.7554/eLife.57613](https://doi.org/10.7554/eLife.57613)
