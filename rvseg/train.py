#!/usr/bin/env python

from __future__ import division, print_function

import argparse

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from rvseg import dataset, model, loss, opts


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
    args = opts.parse_arguments()

    train_generator, train_steps_per_epoch, \
        val_generator, val_steps_per_epoch = dataset.create_generators(
            args.data_dir, args.batch_size, args.validation_split)

    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, maps = images.shape
    _, _, _, classes = masks.shape

    m = model.u_net(height, width, maps,
                    features=args.features,
                    depth=args.depth,
                    classes=classes,
                    temperature=args.temperature,
                    padding=args.padding,
                    batch_norm=args.batch_norm)

    m.summary()

    if args.load_weights:
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
        lossfunc = 'categorical_crossentropy'
    elif args.loss == 'dice':
        def lossfunc(y_true, y_pred):
            return loss.sorensen_dice_loss(y_true, y_pred, [0.1, 0.9])
    elif args.loss == 'jaccard':
        def lossfunc(y_true, y_pred):
            return loss.jaccard_loss(y_true, y_pred, [0.1, 0.9])
    else:
        raise Exception("Unknown loss ({})".format(args.loss))

    def dice(y_true, y_pred):
        batch_dice_coefs = loss.sorensen_dice(y_true, y_pred, axis=[1, 2])
        dice_coefs = K.mean(batch_dice_coefs, axis=0)
        return dice_coefs[1]

    def jaccard(y_true, y_pred):
        batch_jaccard_coefs = loss.jaccard(y_true, y_pred, axis=[1, 2])
        jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
        return jaccard_coefs[1]

    metrics = ['accuracy', dice, jaccard]

    m.compile(optimizer=optimizer, loss=lossfunc, metrics=metrics)

    # automatic saving of model during training
    if args.checkpoint:
        if args.loss == 'pixel':
            filepath="weights-{epoch:02d}-{val_acc:.4f}.hdf5"
            monitor = 'val_acc'
            mode = 'max'
        elif args.loss == 'dice':
            filepath="weights-{epoch:02d}-{val_dice:.4f}.hdf5"
            monitor='val_dice'
            mode = 'max'
        elif args.loss == 'jaccard':
            filepath="weights-{epoch:02d}-{val_jaccard:.4f}.hdf5"
            monitor='val_jaccard'
            mode = 'max'
        checkpoint = ModelCheckpoint(
            filepath, monitor=monitor, verbose=1,
            save_best_only=True, mode=mode)
        callbacks = [checkpoint]
    else:
        callbacks = []

    # train
    m.fit_generator(train_generator,
                    epochs=args.epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=2)

    m.save(args.outfile)

if __name__ == '__main__':
    train()
