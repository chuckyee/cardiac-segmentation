#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
from rvseg import opts, patient, dataset, models


def sorensen_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return 2*intersection / (np.sum(y_true) + np.sum(y_pred))

def jaccard(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return intersection / union

def compute_statistics(model, generator, steps_per_epoch):
    dices = []
    jaccards = []
    for i in range(steps_per_epoch):
        images, masks_true = next(generator)
        masks_pred = model.predict(images)
        for mask_true, mask_pred in zip(masks_true, masks_pred):
            y_true = mask_true[:,:,1].astype('uint8')
            y_pred = np.round(mask_pred[:,:,1]).astype('uint8')
            dices.append(sorensen_dice(y_true, y_pred))
            jaccards.append(jaccard(y_true, y_pred))
    print("Dice:    {:.3f} ({:.3f})".format(np.mean(dices), np.std(dices)))
    print("Jaccard: {:.3f} ({:.3f})".format(np.mean(jaccards), np.std(jaccards)))
    return dices, jaccards

def main():
    args = opts.parse_arguments()

    print("Loading dataset...")
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

    print("Building model...")
    string_to_model = {
        "unet": models.unet,
        "dilated-unet": models.dilated_unet,
        "dilated-densenet": models.dilated_densenet,
    }
    model = string_to_model[args.model]

    m = model(height=height, width=width, channels=channels, classes=classes,
              features=args.features, depth=args.depth, padding=args.padding,
              temperature=args.temperature, batchnorm=args.batchnorm,
              dropout=args.dropout)

    m.load_weights(args.load_weights)

    print("Training Set:")
    train_dice, train_jaccard = compute_statistics(m, train_generator, train_steps_per_epoch)
    print()
    print("Validation Set:")
    val_dice, val_jaccard = compute_statistics(m, val_generator, val_steps_per_epoch)

    if args.outfile:
        train_data = np.asarray([train_dice, train_jaccard]).T
        val_data = np.asarray([val_dice, val_jaccard]).T
        np.savetxt(args.outfile + ".train", train_data)
        np.savetxt(args.outfile + ".val", val_data)



if __name__ == '__main__':
    main()
