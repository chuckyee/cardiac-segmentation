# Cardiac MRI Segmentation

![build status](https://travis-ci.org/chuckyee/cardiac-segmentation.svg?branch=master)

This repository contains code and models to identify the right ventricle in cardiac MRI images. The [project](http://ai-on.org/projects/cardiac-mri-segmentation.html) is one of the open calls for research from Francois Chollet's [AI Open Network](http://ai-on.org/).

## Problem description

From the call for research: 

> Develop a system capable of automatic segmentation of the right ventricle in
> images from cardiac magnetic resonance imaging (MRI) datasets. Until now,
> this has been mostly handled by classical image processing methods. Modern
> deep learning techniques have the potential to provide a more reliable,
> fully-automated solution.
>
> A recent [Kaggle
> challenge](https://www.kaggle.com/c/second-annual-data-science-bowl) focused
> on measuring the volume of the left ventricle from MRI images. Deep learning
> techniques proved very effective, and some of the top entries from this
> challenge can provide inspiration for right-ventricle segmentation. Note that
> right-ventricle segmentation is likely to be a harder problem due to the more
> complex geometry of the right ventricle. Several challenges exist in the
> automatic segmentation of the right ventricle: presence of trabeculations in
> the cavity with signal intensities similar to that of the myocardium; the
> complex crescent shape of the RV, which varies from the base to the apex;
> difficulty in segmenting the apical image slices; considerable variability in
> shape and intensity of the chamber among subjects, notably in pathological
> cases, etc.

## Why this problem matters

Segmentation of the right ventricle is the first step in estimating the
ventricular volume, a key diagnostic for heart disease. Again from the call for
research:

> Cardiac MRI is routinely being used for the evaluation of the function and
> structure of the cardiovascular system. The obtained magnetic resonance
> images of patients are inspected both visually and quantitatively by
> clinicians to extract important information about heart function.
> Segmentation of the heart chambers, such as the right ventricle, in cardiac
> magnetic resonance images is an essential step for the computation of
> clinical indices such as ventricular volume and ejection fraction (note: you
> could also try to directly predict ventricular volume, although it would be
> more difficult).
>
> Manual delineation by experts is currently the standard clinical practice for
> right ventricle segmentation. However, manual segmentation is tedious, time
> consuming and prone to intra and inter-observer variability. Therefore, it is
> necessary to reproducibly automate this task to accelerate and facilitate the
> process of diagnosis and follow-up.

## How to measure success

We use the Dice coefficient to compare the pixel-wise agreement between the
predicted segmentation and the corresponding ground truth.

## Dataset

The dataset (accessible
[here](http://www.litislab.fr/?sub_project=how-to-download-the-data)) contains
3940 MRI images, with 243 of them labeled from which the training and
validation set is drawn. The small size of labeled data is common in medical
contexts and will require the extensive use of data augmentation. Additionally,
the accuracy of the ground truths are dependent on the skill of the labeling
physician.

![pixel-stats](/figures/pixel-statistics.png)

## References

* Overview of state-of-the-art of CNN applications in medical segmentation [link](https://arxiv.org/abs/1701.03056).
* Right Ventricle Segmentation From Cardiac MRI: A Collation Study
* A survey of shaped-based registration and segmentation techniques for cardiac images
* Right ventricular segmentation in cardiac MRI with moving mesh correspondences
* Segmentation of RV in 4D cardiac mr volumes using region-merging graph cuts

Note: this entire library is written with the Tensorflow backend in mind --
(batch, height, width, channels) ordered is assumed and is not portable to
Theano.