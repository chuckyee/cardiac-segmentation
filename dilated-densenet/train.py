from __future__ import print_function, division

import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from rvseg import dataset, loss, opts
from densenet import DenseNet


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
    model = DenseNet()
    print(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        for step in range(train_steps_per_epoch):
            images, masks = next(train_generator)
            images = Variable(torch.FloatTensor(images))
            masks = Variable(torch.FloatTensor(masks))
            masks_pred = model(images)
            loss = criterion(masks_pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # check validation error

if __name__ == '__main__':
    train()
