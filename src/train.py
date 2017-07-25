#!/usr/bin/env python

from __future__ import division, print_function

from keras import losses, optimizers, utils

import glob
import patient
import model


def main():
    learning_rate = 0.01
    momentum = 0.99
    decay = 0.0
    epochs = 100
    validation_split = 0.2
    padding = 'same'
    features = 32
    depth = 3
    classes = 2

    import numpy as np
    patient_dirs = glob.glob("/home/paperspace/Developer/datasets/RVSC/TrainingSet/patient*")[:1]
    images = []
    masks = []
    for patient_dir in patient_dirs:
        p = patient.PatientData(patient_dir)
        images += p.images
        masks += p.endocardium_masks
    images = np.asarray(images)[:,:,:,None]
    masks = np.asarray(masks) // 255
    print(images.shape, masks.shape)
    print(set(masks.flatten()))
    dims = masks.shape
    masks = utils.to_categorical(masks).reshape(*dims, classes)
    print(masks.shape)

    _, height, width, _ = images.shape
    print(height, width)
    m = model.u_net(height, width, features, depth, classes, padding)

    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)

    m.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    m.fit(images, masks, epochs=epochs, validation_split=validation_split)


if __name__ == '__main__':
    main()
