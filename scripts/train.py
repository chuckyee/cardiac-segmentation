#!/usr/bin/env python

from __future__ import division, print_function

import os
import argparse
import logging

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from rvseg import dataset, models, loss, opts


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
    logging.basicConfig(level=logging.INFO)

    args = opts.parse_arguments()

    logging.info("Loading dataset...")
    augmentation_args = {
        'rotation_range': args.rotation_range,
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'shear_range': args.shear_range,
        'zoom_range': args.zoom_range,
        'fill_mode' : args.fill_mode,
        'alpha': args.alpha,
        'sigma': args.sigma,
    }
    train_generator, train_steps_per_epoch, \
        val_generator, val_steps_per_epoch = dataset.create_generators(
            args.datadir, args.batch_size,
            validation_split=args.validation_split,
            mask=args.classes,
            shuffle_train_val=args.shuffle_train_val,
            shuffle=args.shuffle,
            seed=args.seed,
            normalize_images=args.normalize,
            augment_training=args.augment_training,
            augment_validation=args.augment_validation,
            augmentation_args=augmentation_args)

    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, channels = images.shape
    _, _, _, classes = masks.shape

    logging.info("Building model...")
    string_to_model = {
        "unet": models.unet,
        "dilated-unet": models.dilated_unet,
        "dilated-densenet": models.dilated_densenet,
        "dilated-densenet2": models.dilated_densenet2,
        "dilated-densenet3": models.dilated_densenet3,
    }
    model = string_to_model[args.model]
    m = model(height=height, width=width, channels=channels, classes=classes,
              features=args.features, depth=args.depth, padding=args.padding,
              temperature=args.temperature, batchnorm=args.batchnorm,
              dropout=args.dropout)

    m.summary()

    if args.load_weights:
        logging.info("Loading saved weights from file: {}".format(args.load_weights))
        m.load_weights(args.load_weights)

    # instantiate optimizer, and only keep args that have been set
    # (not all optimizers have args like `momentum' or `decay')
    optimizer_args = {
        'lr':       args.learning_rate,
        'momentum': args.momentum,
        'decay':    args.decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer(args.optimizer, optimizer_args)

    # select loss function: pixel-wise crossentropy, soft dice or soft
    # jaccard coefficient
    if args.loss == 'pixel':
        def lossfunc(y_true, y_pred):
            return loss.weighted_categorical_crossentropy(
                y_true, y_pred, args.loss_weights)
    elif args.loss == 'dice':
        def lossfunc(y_true, y_pred):
            return loss.sorensen_dice_loss(y_true, y_pred, args.loss_weights)
    elif args.loss == 'jaccard':
        def lossfunc(y_true, y_pred):
            return loss.jaccard_loss(y_true, y_pred, args.loss_weights)
    else:
        raise Exception("Unknown loss ({})".format(args.loss))

    def dice(y_true, y_pred):
        batch_dice_coefs = loss.sorensen_dice(y_true, y_pred, axis=[1, 2])
        dice_coefs = K.mean(batch_dice_coefs, axis=0)
        return dice_coefs[1]    # HACK for 2-class case

    def jaccard(y_true, y_pred):
        batch_jaccard_coefs = loss.jaccard(y_true, y_pred, axis=[1, 2])
        jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
        return jaccard_coefs[1] # HACK for 2-class case

    metrics = ['accuracy', dice, jaccard]

    m.compile(optimizer=optimizer, loss=lossfunc, metrics=metrics)

    # automatic saving of model during training
    if args.checkpoint:
        if args.loss == 'pixel':
            filepath = os.path.join(
                args.outdir, "weights-{epoch:02d}-{val_acc:.4f}.hdf5")
            monitor = 'val_acc'
            mode = 'max'
        elif args.loss == 'dice':
            filepath = os.path.join(
                args.outdir, "weights-{epoch:02d}-{val_dice:.4f}.hdf5")
            monitor='val_dice'
            mode = 'max'
        elif args.loss == 'jaccard':
            filepath = os.path.join(
                args.outdir, "weights-{epoch:02d}-{val_jaccard:.4f}.hdf5")
            monitor='val_jaccard'
            mode = 'max'
        checkpoint = ModelCheckpoint(
            filepath, monitor=monitor, verbose=1,
            save_best_only=True, mode=mode)
        callbacks = [checkpoint]
    else:
        callbacks = []

    # train
    logging.info("Begin training.")
    m.fit_generator(train_generator,
                    epochs=args.epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=2)

    m.save(os.path.join(args.outdir, args.outfile))

if __name__ == '__main__':
    train()
