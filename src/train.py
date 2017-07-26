#!/usr/bin/env python

from __future__ import division, print_function

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import argparse
import glob

import patient
import model


def handle_arguments():
    parser = argparse.ArgumentParser(description="Train U-Net from right ventricle MRI images.")
    parser.add_argument('-d', '--data_dir', default='.', help="Directory containing patientXX/ directories")
    parser.add_argument('-o', '--outdir', default='.', help="Directory to write output data")
    parser.add_argument('-s', '--split', default=20, type=int, help='Percentage of patients used for test set')
    parser.add_argument('--optimizer')
    parser.add_argument('--optimizer')
    parser.add_argument('--optimizer')
    args = parser.parse_args()
    main(args)

smooth = 100.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def main():
    learning_rate = 0.01
    momentum = 0.99
    decay = 0.0
    epochs = 100
    validation_split = 0.2
    padding = 'same'
    maps = 1
    features = 32
    depth = 3
    classes = 2

    import numpy as np
    patient_dirs = glob.glob("/home/paperspace/Developer/datasets/RVSC/TrainingSet/patient*")
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
    for i, mask in enumerate(masks[-10:]):
        np.savetxt("mask{}-0.txt".format(i), mask[:,:,0])
        np.savetxt("mask{}-1.txt".format(i), mask[:,:,1])

    _, height, width, _ = images.shape
    print(height, width)
    m = model.u_net(height, width, maps, features, depth, classes, padding)

    # optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)
    optimizer = optimizers.Adam(lr=1e-3)

    m.load_weights("best/weights-improvement-89-0.98454.hdf5")
    m.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # m.compile(optimizer=optimizer,
    #           loss=dice_coef_loss,
    #           metrics=[dice_coef])

    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # filepath="weights-improvement-{epoch:02d}-{val_dice_coef:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
    callbacks = [checkpoint]
    m.fit(images, masks, epochs=epochs, validation_split=validation_split,
          callbacks=callbacks)

    predictions = m.predict(images[-10:])
    for i, prediction in enumerate(predictions):
        np.savetxt("predict{}-0.txt".format(i), prediction[:,:,0])
        np.savetxt("predict{}-1.txt".format(i), prediction[:,:,1])

def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if optimizer_name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(name))

    return optimizers[optimizer_name](**optimizer_args)

def train():
    train_generator, val_generator = load_data(args.data_dir)

    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, maps = images.shape

    m = model.u_net(height, width, maps,
                    features=args.features,
                    depth=args.depth,
                    classes=args.classes,
                    padding=args.padding)

    if args.saved_weights:
        m.load_weights(args.saved_weights)

    # instantiate optimizer, and only keep args that have been set (not all
    # optimizers have args like `momentum' or `decay'
    optimizer_args = {
        'lr':       args.learning_rate,
        'momentum': args.momentum,
        'decay':    args.decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer(args.optimizer, optimizer_args)

    # select loss function (pixel-wise crossentropy or dice coefficient)
    if args.loss == 'pixel':
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    elif args.loss == 'dice':
        loss = dice_coef_loss
        metrics = [dice_coef]
    else:
        raise Exception("Unknown loss ({})".format(args.loss))

    m.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # automatic saving of model during training
    if args.checkpoint:
        if args.loss == 'pixel':
            filepath="weights-{epoch:02d}-{val_dice_coef:.4f}.hdf5"
            monitor='val_dice_coef'
        else:
            filepath="weights-{epoch:02d}-{val_acc:.4f}.hdf5"
            monitor = 'val_acc'
        checkpoint = ModelCheckpoint(
            filepath, monitor=monitor, verbose=1,
            save_best_only=True, mode='max')
        callbacks = [checkpoint]
    else:
        callbacks = []

    # train
    m.fit(images, masks, epochs=args.epochs,
          validation_split=args.validation_split,
          callbacks=callbacks)

    m.save(args.output)

if __name__ == '__main__':
    train()
