from __future__ import division, print_function

import os
import glob
import numpy as np
from math import ceil

from keras import utils
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator

import patient


def load_images(data_dir):
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

    return images, masks

def random_elastic_deformation(x, a, sigma):
    return x

class Iterator(object):
    def __init__(self, images, masks, batch_size):
        self.images = images
        self.masks = masks
        self.i = 0
        self.batch_size = batch_size
        augmentation_args = {
            'rotation_range': 180,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.05,
            'fill_mode' : 'reflect',
        }
        self.idg = ImageDataGenerator(**augmentation_args)

    def __next__(self):
        return self.next()

    def next(self):
        start = self.i
        end = min(start + self.batch_size, len(self.images))
        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
        augmented_images = []
        augmented_masks = []
        images_masks = zip(self.images[start:end], self.masks[start:end])
        for image, mask in images_masks:
            _, _, channels = image.shape
            stacked = np.concatenate((image, mask), axis=2)
            augmented = self.idg.random_transform(stacked)
            augmented = random_elastic_deformation(augmented, 0, 0)
            augmented_images.append(augmented[:,:,:channels])
            augmented_masks.append(augmented[:,:,channels:])
        return np.asarray(augmented_images), np.asarray(augmented_masks)

def create_generators(data_dir, batch_size, validation_split=0.0):
    images, masks = load_images(data_dir)

    # split out last %(validation_split) of images as validation set
    split_index = int((1-validation_split) * len(images))

    # Augment images and masks from training set
    train_generator = Iterator(images[:split_index], masks[:split_index],
                               batch_size)

    train_steps_per_epoch = ceil(split_index / batch_size)

    # Do not augment validation data
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


def create_generators_keras(data_dir, batch_size, validation_split=0.0):
    # Basic image / mask loader using keras; no image augmentation
    images, masks = load_images(data_dir)

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
