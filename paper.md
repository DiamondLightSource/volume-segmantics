---
title: 'Volume Segmantics: A Python package for semantic segmentation of volumetric data using pre-trained PyTorch deep learning models'
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

Segmentation of volumetric (3-dimensional (3D)) images is a widely used technique 
that allows interpretation and quantification of experimental data collected 
using a number of techniques (e.g. Computed Tomography (CT), Magnetic Resonance 
Imaging (MRI), Electron Tomography (ET)). Although the idea of semantic 
segmentation is a relatively simple one, giving each pixel a label that defines 
what it represents (e.g cranial bone versus brain tissue), due to the subjective 
and laborious nature of the manual labelling task coupled with the huge size of the 
data (multi-GB files containing billions of pixels) this process is a bottleneck 
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
encoders which have been pre-trained on huge databases such as ImageNet (ref1). 
This provides two main advantages, namely (i) it to reduces the time and 
resource taken to train models - only fine-tuning is required and (ii) it prevents
over-fitting the models when training on small datasets. These models of various 
architectures are included from the (segmentation models pytorch ref) repository.
In order to increase the accuracy of the models, augmentations of the data are
made during training, both via 'slicing' the 3D data in planes perpendicular to 
the three orthogonal axes $((x, y), (x, z), (y, z))$ and by using the library 
Albumentations (insert ref). Additionally, user configuration for training is 
kept to a minimum by starting with a reliable default set of parameters and by 
automatically choosing the model learning rate.

Even though these 2D models are quicker to train and require fewer computational 
resources than their 3D counterparts (ref gas hydrates benchmarking), when 
predicting a segmentation for a volume, the lack of 2D context available to these 
models can lead to striping artifacts in the 3D output, especially when viewed 
in planes other than the one used for prediction. To overcome this, a multi-axis 
prediction method is used, and the multiple predictions are merged by using 
maximum probability voting. It is hoped that in the future other merging techniques 
will be included such as those in (ref).

During development of `Volume Segmantics` pre-trained U-Net models have been 
given additional fine-tuning on small amounts of annotated data in order to 
investigate the structures that interface maternal and fetal blood volumes in 
human placental tissue (ref). In this study, expert annotation of volumes of 
size $256^3$ and $384^3$ were sufficient to create two models that gave accurate 
segmentation of two much larger synchrotron X-ray CT (SXCT) datasets 
$(2520 \times 2520 \times 2120 px)$. In a completely different context, SXCT 
datasets collected on a soil system in which methane bubbles were forming in 
a system containing sand and brine were used to study the utility of
a 2D U-Net in comparison to its 3D counterpart (ref Gas hydrates). Here, the 
training data ranged in size from $384^3$ pixels to $572^3$. As well as requiring 
less time to train than a 3D U-Net, the pre-trained 2D network provided more 
accurate segmentation results. 

The API provided with the package allows segmentation models to be trained and 
used in other contexts. For example, `Volume Segmantics` is currently being 
integrated into `SuRVoS2`, a client-server application with a GUI for annotating 
volumetric data in order to train machine learning models via a series of 'scribbles' 
applied to the data. It is hoped that scientists using our synchrotron facility 
will be able to easily train and use their own deep learning 
models to perform segmentation during their time here and when back at their own 
institutions and that other scientists will use and extend the package for their 
own purposes. 

# Acknowledgements

We would like to acknowledge helpful discussions with Avery Pennington, Luis Perdigao 
and Michele Darrow during the development of this project.

# References