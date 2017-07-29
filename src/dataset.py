from __future__ import division, print_function

import os
import glob
import numpy as np
from math import ceil

from keras import utils
from keras.preprocessing.image import ImageDataGenerator

import patient


def elastic_deformation(a, sigma):
    augmentation_args = {
        'rotation_range': 10,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.1,
        'zoom_range': 0.1,
        'fill_mode' : 'reflect',
    }

def create_generators(data_dir, batch_size, validation_split=0.0):
    glob_search = os.path.join(data_dir, "patient*")
    patient_dirs = glob.glob(glob_search)

    # load all images into memory (dataset is small)
    images = []
    masks = []
    for patient_dir in patient_dirs:
        p = patient.PatientData(patient_dir)
        images += p.images
        masks += p.endocardium_masks

    # reshape to account for channel dimension
    images = np.asarray(images)[:,:,:,None]
    masks = np.asarray(masks) // 255 # convert grayscale mask to {0, 1} mask

    # one-hot encode masks
    dims = masks.shape
    classes = len(set(masks[0].flatten())) # get num classes from first image
    masks = utils.to_categorical(masks).reshape(*dims, classes)

    # split out last %(validation_split) of images as validation set
    split_index = int((1-validation_split) * len(images))

    train_generator = ImageDataGenerator().flow(
        images[:split_index],
        masks[:split_index],
        batch_size=batch_size)

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        val_generator = ImageDataGenerator().flow(
            images[split_index:],
            masks[split_index:],
            batch_size=batch_size)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch,
            val_generator, val_steps_per_epoch)
