---
title: 'Volume Segmantics: A Python Package for Semantic Segmentation of Volumetric Data Using Pre-trained PyTorch Deep Learning Models'
tags:
  - Python
  - segmentation
  - deep learning
  - volumetric
  - images
  - pre-trained
authors:
  - name: Oliver N. F. King
    orcid: 0000-0002-6152-7207
    corresponding: true
    affiliation: 1 
  - name: Dimitrios Bellos
    orcid: 0000-0002-8015-3191
    affiliation: 2
  - name: Mark Basham
    orcid: 0000-0002-8438-1415
    affiliation: "1, 2"
affiliations:
 - name: Diamond Light Source Ltd., Harwell Science and Innovation Campus, Didcot, Oxfordshire, UK
   index: 1
 - name: Rosalind Franklin Institute, Harwell Science and Innovation Campus, Didcot, Oxfordshire, UK
   index: 2
date: 20 July 2022
bibliography: paper.bib

---

# Summary

Segmentation of 3-dimensional (3D, volumetric) images is a widely used technique 
that allows interpretation and quantification of experimental data collected 
using a number of techniques (for example, Computed Tomography (CT), Magnetic Resonance 
Imaging (MRI), Electron Tomography (ET)). Although the idea of semantic 
segmentation is a relatively simple one, giving each pixel a label that defines 
what it represents (e.g cranial bone versus brain tissue); due to the subjective 
and laborious nature of the manual labelling task coupled with the huge size of the 
data (multi-GB files containing billions of pixels) this process is often a bottleneck 
in imaging workflows. In recent years, deep learning has brought models capable of 
fast and accurate interpretation of image data into the toolbox available to 
scientists. These models are often trained on large image datasets that have been 
annotated at great expense. In many cases however, scientists working on novel 
samples and using new imaging techniques do not yet have access to large 
stores of annotated data. To overcome this issue, simple software tools that 
allow the scientific community to create segmentation models using relatively 
small amounts of training data are required. `Volume Segmantics` is a Python 
package that provides a command line interface (CLI) as well as an Application 
Programming Interface (API) for training 2-dimensional (2D)
PyTorch deep learning models on small amounts of annotated 3D image 
data.  The package also enables applying these models to new (often much larger) 
3D datasets to speed up the process of semantic segmentation.


# Statement of need

`Volume Segmantics` harnesses the availability of 2-dimensional 
encoders which have been pre-trained on huge databases such as ImageNet 
[@russakovsky_imagenet_2015]. This provides two main advantages, namely (i) 
it reduces the time and resource taken to train models &mdash; only fine-tuning is 
required and (ii) it prevents over-fitting the models when training on small 
datasets. These models of various architectures are included from the 
`segmentation-models-pytorch` repository [@Yakubovskiy:2019].
In order to increase the accuracy of the models, augmentations of the data are
made during training, both via 'slicing' the 3D data in planes perpendicular to 
the three orthogonal axes $((x, y), (x, z), (y, z))$ and by using the library 
`Albumentations` [@buslaev_albumentations_2020]. Additionally, user configuration 
for training is kept to a minimum by starting with a reliable default set of 
parameters and by automatically choosing the model learning rate. If adjustments 
to model architecture, encoder type, loss function or training epochs are required; 
these can be made by editing a YAML file.

Even though these 2D models are quicker to train and require fewer computational 
resources than their 3D counterparts [@alvarez-borges_u-net_2022], when 
predicting a segmentation for a volume, the lack of 3D context available to these 
models can lead to striping artifacts in the 3D output, especially when viewed 
in planes other than the one used for prediction. To overcome this, a multi-axis 
prediction method is used, and the multiple predictions are merged by using 
maximum probability voting. It is hoped that in the future other merging techniques 
will be included such as fusion models [@perslev_one_2019]. A schematic of the training and prediction processes performed by the `Volume Segmantics` package can be seen in \autoref{fig:schematic}.

![A schematic diagram showing the model training and segmentation prediction processes performed by the `Volume Segmantics` package.\label{fig:schematic}](schematic_hig_res_crop.png)

## State of the field

Currently there are a number of other software implementations available for segmentation of 3D data. Some of these also use 2d networks and combine prediction outputs to 3D, for example the `Multi-Planar U-Net` package [@perslev_github_2019] and the `CTSegNet` package [@tekawade_github_2020]. However, neither of these packages allows the use of pre-trained encoders or multiple model architectures. In addition, the general purpose `pytorch-3dunet` package [@wolny_github_2019] exists to allow training a 3D U-Net on image data, again without the time and resource advantages of pre-trained 2D networks. 

In the field of connectomics, several packages [@lee_deepem_2018; @lin2021pytorch; @urakubo_uni-em_2019; @wu_neutorch_2021] enable the segmentation of structures within the brain, often from electron microscopy data, these could in principle be used in a subject and method-agnostic manner similarly to `volume-segmantics`. One member of this set, the package `pytorch-connectomics`, [@lin_pth_connec_github_2019] allows training of 2D and 3D networks for segmentation of 3D image data as well as customisable strategies for data augmentation and multi-task and semi-supervised learning. Despite the versatility of this software, its deliberate focus on connectomics which is essential for effectiveness in this complex field, mean that there are added levels of complexity for the generalist user. This specialisation also means that there is only one pre-trained 2D model architecture available, and some of the configuration and command-line options are context specific.

## Real-world usage

During development of `Volume Segmantics`, the software was used to fine-tune pre-trained U-Net models on small amounts of annotated data in order to 
investigate the structures that interface maternal and fetal blood volumes in 
human placental tissue [@tun_massively_2021]. In this study, expert annotation of 
volumes of size $256^3$ and $384^3$ were sufficient to create two models that gave 
accurate segmentation of two much larger synchrotron X-ray CT (SXCT) datasets 
$(2520 \times 2520 \times 2120 pixels)$. In a completely different context, SXCT 
datasets were collected on a soil system in which methane bubbles were forming in 
brine amongst sand particles. The utility of a pre-trained 2D U-Net was investigated 
to segment these variable-contrast image volumes in comparison to a 3D U-Net with 
no prior training [@alvarez-borges_u-net_2022]. In this case, the training data 
ranged in size from $384^3$ pixels to $572^3$. As well as requiring less time to 
train than a 3D U-Net, the pre-trained 2D network provided more accurate segmentation 
results. 

## The API

The API provided with the package allows segmentation models to be trained and 
used in other contexts. For example, `Volume Segmantics` has recently been 
integrated into `SuRVoS2` [@survos2:2018; @pennington_survos_2022], a client-server 
application with a GUI for annotating volumetric data. SuRVoS2 can be used to create the 
initial small region of interest (ROI) labels needed by `Volume Segmantics`, this is achieved by using machine learning models (e.g. random forests) which are trained through 'scribbles' drawn on the data. It is hoped that scientists using our 
synchrotron facility and beyond will be able to train and use their own deep 
learning models using this interface to the library. These models can then be 
used to segment data during their time here and also when back at their home 
institution. In addition, it is hoped that the scientific community will use and 
extend `Volume Segmantics` for their own purposes. 

# Acknowledgements

We would like to acknowledge helpful discussions with Avery Pennington, Sharif Ahmed, 
Fernando Alvarez-Borges and Michele Darrow during the development of 
this project. Additional thanks to Luis Perdigao for inspiration and icons for the schematic figure. 

# References
