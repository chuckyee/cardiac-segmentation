from __future__ import division, print_function

import os
import glob
import numpy as np
from math import ceil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from keras import utils
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator

from . import patient


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

def random_elastic_deformation(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width, channels = image.shape

    dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = (np.repeat(np.ravel(x+dx), 3),
               np.repeat(np.ravel(y+dy), 3),
               np.tile([0,1,2], height*width))

    return map_coordinates(image, indices, order=1).reshape((height, width, channels))

class Iterator(object):
    def __init__(self, images, masks, batch_size, augmentation_args={}):
        self.images = images
        self.masks = masks
        self.i = 0
        self.batch_size = batch_size
        self.idg = ImageDataGenerator(**augmentation_args)
        self.alpha = 500
        self.sigma = 20

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
            augmented = random_elastic_deformation(
                augmented, self.alpha, self.sigma)
            augmented_images.append(augmented[:,:,:channels])
            augmented_masks.append(np.round(augmented[:,:,channels:]))
        return np.asarray(augmented_images), np.asarray(augmented_masks)

def create_generators(data_dir, batch_size, validation_split=0.0,
                      augment_training=True, augment_validation=False,
                      augmentation_args={}):
    images, masks = load_images(data_dir)

    # split out last %(validation_split) of images as validation set
    split_index = int((1-validation_split) * len(images))

    if augment_training:
        train_generator = Iterator(
            images[:split_index], masks[:split_index],
            batch_size, augmentation_args)
    else:
        train_generator = ImageDataGenerator().flow(
            images[split_index:], masks[split_index:], batch_size=batch_size)

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        if augment_validation:
            val_generator = Iterator(
                images[split_index:], masks[split_index:],
                batch_size, augmentation_args)
        else:
            val_generator = ImageDataGenerator().flow(
                images[split_index:], masks[split_index:],
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
