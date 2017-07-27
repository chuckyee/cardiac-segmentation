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
import dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train U-Net to segment right ventricles from cardiac MRI images.")
    parser.add_argument('-d', '--data_dir', default='.', help="Directory containing patientXX/ directories")
    parser.add_argument('-o', '--outfile', default='weights-final.hdf5', help="Directory to write output data")
    parser.add_argument('--features', default=64, type=int, help="Number of features maps after first convolutional layer.")
    parser.add_argument('--depth', default=4, type=int, help="Number of downsampled convolutional blocks.")
    # parser.add_argument('--classes')
    parser.add_argument('--padding', default='same', help="Padding in convolutional layers. Either `same' or `valid'.")
    parser.add_argument('--saved_weights', help="Begin training from model weights saved in specified file.")
    parser.add_argument('--learning_rate', default=None, help="Optimizer learning rate.")
    parser.add_argument('--momentum', default=None, type=float, help="Momentum for SGD optimizer.")
    parser.add_argument('--decay', default=None, type=float, help="Learning rate decay (not applicable for nadam).")
    parser.add_argument('--optimizer', default='adam', help="Optimizer: sgd, rmsprop, adagrad, adadelta, adam, adamax, or nadam.")
    parser.add_argument('--loss', default='pixel', help="Loss function: `pixel' for pixel-wise cross entropy, `dice' for dice coefficient.")
    parser.add_argument('--checkpoint', action='store_true', help="Write model weights after each epoch if validation accuracy improves.")
    parser.add_argument('-e', '--epochs', default=20, type=int, help="Number of epochs to train.")
    parser.add_argument('-b', '--batch_size', default=32, type=int, help="Mini-batch size for training.")
    parser.add_argument('--validation_split', default=0.2, type=float, help="Percentage of training data to hold out for validation.")
    args = parser.parse_args()
    return args

def dice_coef(y_true, y_pred):
    # input tensors have shape (batch_size, height, width, classes)
    smooth = 1.
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f)
    # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # y_true_flat = K.flatten(y_true)
    # y_pred_flat = K.flatten(y_pred)
    # intersection = K.sum(y_true_flat * y_pred_flat)
    # dice = (2. * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)
    # return dice

    # y_pred_int = y_pred
    # intersection = K.sum(y_true * y_pred_int)
    # area1 = K.sum(y_true)
    # area2 = K.sum(y_pred_int)
    # dice = (2 * intersection + smooth) / (area1 + area2 + smooth)
    # return dice

    y_pred_int = y_pred
    intersection = K.sum(y_true * y_pred_int, axis=[0, 1, 2])
    area1 = K.sum(y_true)
    area2 = K.sum(y_pred_int)
    dice = (2 * intersection + smooth) / (area1 + area2 + smooth)
    return dice[0]

    # _, _, _, classes = K.int_shape(y_true)
    # dice_coefs = []
    # for i in range(classes):
    #     y_true_class = y_true[:,:,:,i]
    #     y_pred_class = K.round(y_pred[:,:,:,i])
    #     intersection = K.sum(y_true_class * y_pred_class)
    #     area1 = K.sum(y_true_class)
    #     area2 = K.sum(y_pred_class)
    #     dice_coefs.append((2 * intersection + smooth) / (area1 + area2 + smooth))
    # return dice_coefs[1]

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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
    args = parse_arguments()

    train_generator, val_generator = dataset.create_generators(
        args.data_dir, args.batch_size, args.validation_split)

    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, maps = images.shape
    _, _, _, classes = masks.shape

    m = model.u_net(height, width, maps,
                    features=args.features,
                    depth=args.depth,
                    classes=classes,
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
            filepath="weights-{epoch:02d}-{val_acc:.4f}.hdf5"
            monitor = 'val_acc'
        else:
            filepath="weights-{epoch:02d}-{val_dice_coef:.4f}.hdf5"
            monitor='val_dice_coef'
        checkpoint = ModelCheckpoint(
            filepath, monitor=monitor, verbose=1,
            save_best_only=True, mode='max')
        callbacks = [checkpoint]
    else:
        callbacks = []

    # train
    m.fit_generator(train_generator,
                    epochs=args.epochs,
                    steps_per_epoch=8,
                    validation_data=val_generator,
                    validation_steps=20,
                    callbacks=callbacks)

    m.save(args.outfile)

if __name__ == '__main__':
    train()
