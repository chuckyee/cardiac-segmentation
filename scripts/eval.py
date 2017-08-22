#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from rvseg import opts, patient, dataset, models


def save_image(figname, image, mask_true, mask_pred, alpha=0.3):
    cmap = plt.cm.gray
    plt.figure(figsize=(12, 3.75))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.imshow(mask_pred, cmap=cmap, alpha=alpha)
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.imshow(mask_true, cmap=cmap, alpha=alpha)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def sorensen_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return 2*intersection / (np.sum(y_true) + np.sum(y_pred))

def jaccard(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return intersection / union

def compute_statistics(model, generator, steps_per_epoch, return_images=False):
    dices = []
    jaccards = []
    predictions = []
    for i in range(steps_per_epoch):
        images, masks_true = next(generator)
        # Normally: masks_pred = model.predict(images)
        # But dilated densenet cannot handle large batch size
        masks_pred = np.concatenate([model.predict(image[None,:,:,:]) for image in images])
        for mask_true, mask_pred in zip(masks_true, masks_pred):
            y_true = mask_true[:,:,1].astype('uint8')
            y_pred = np.round(mask_pred[:,:,1]).astype('uint8')
            dices.append(sorensen_dice(y_true, y_pred))
            jaccards.append(jaccard(y_true, y_pred))
        if return_images:
            for image, mask_true, mask_pred in zip(images, masks_true, masks_pred):
                predictions.append((image[:,:,0], mask_true[:,:,1], mask_pred[:,:,1]))
    print("Dice:    {:.3f} ({:.3f})".format(np.mean(dices), np.std(dices)))
    print("Jaccard: {:.3f} ({:.3f})".format(np.mean(jaccards), np.std(jaccards)))
    return dices, jaccards, predictions

def main():
    # Sort of a hack:
    # args.outfile = file basename to store train / val dice scores
    # args.checkpoint = turns on saving of images
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
        "dilated-densenet2": models.dilated_densenet2,
        "dilated-densenet3": models.dilated_densenet3,
    }
    model = string_to_model[args.model]

    m = model(height=height, width=width, channels=channels, classes=classes,
              features=args.features, depth=args.depth, padding=args.padding,
              temperature=args.temperature, batchnorm=args.batchnorm,
              dropout=args.dropout)

    m.load_weights(args.load_weights)

    print("Training Set:")
    train_dice, train_jaccard, train_images = compute_statistics(
        m, train_generator, train_steps_per_epoch,
        return_images=args.checkpoint)
    print()
    print("Validation Set:")
    val_dice, val_jaccard, val_images = compute_statistics(
        m, val_generator, val_steps_per_epoch,
        return_images=args.checkpoint)

    if args.outfile:
        train_data = np.asarray([train_dice, train_jaccard]).T
        val_data = np.asarray([val_dice, val_jaccard]).T
        np.savetxt(args.outfile + ".train", train_data)
        np.savetxt(args.outfile + ".val", val_data)

    if args.checkpoint:
        print("Saving images...")
        for i,dice in enumerate(train_dice):
            image, mask_true, mask_pred = train_images[i]
            figname = "train-{:03d}-{:.3f}.png".format(i, dice)
            save_image(figname, image, mask_true, np.round(mask_pred))
        for i,dice in enumerate(val_dice):
            image, mask_true, mask_pred = val_images[i]
            figname = "val-{:03d}-{:.3f}.png".format(i, dice)
            save_image(figname, image, mask_true, np.round(mask_pred))

if __name__ == '__main__':
    main()
