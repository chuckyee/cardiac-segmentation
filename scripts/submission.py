#!/usr/bin/env python

from __future__ import division, print_function

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import cv2

from rvseg import opts, patient, dataset, models


def load_patient_images(path, normalize=True):
    p = patient.PatientData(path)

    # reshape to account for channel dimension
    images = np.asarray(p.images, dtype='float64')[:,:,:,None]

    # maybe normalize images
    if normalize:
        dataset.normalize(images, axis=(1,2))

    return images, p.index, p.labeled, p.rotated

def get_contours(mask):
    mask_image = np.where(mask > 0.5, 255, 0).astype('uint8')
    im2, coords, hierarchy = cv2.findContours(mask_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not coords:
        print("No contour detected.")
        coords = np.ones((1, 1, 1, 2), dtype='int')
    if len(coords) > 1:
        print("Multiple contours detected.")
        lengths = [len(coord) for coord in coords]
        coords = [coords[np.argmax(lengths)]]

    coords = np.squeeze(coords[0], axis=(1,))
    coords = np.append(coords, coords[:1], axis=0)

    return coords

def save_image(figname, image, mask_pred, alpha=0.3):
    cmap = plt.cm.gray
    plt.figure(figsize=(8, 3.75))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.imshow(mask_pred, cmap=cmap, alpha=alpha)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def main():
    # Sort of a hack:
    # args.checkpoint = turns on saving of images
    args = opts.parse_arguments()
    args.checkpoint = False     # override for now

    glob_search = os.path.join(args.datadir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directors found in {}".format(data_dir))

    # get image dimensions from first patient
    images, _, _, _ = load_patient_images(patient_dirs[0], args.normalize)
    _, height, width, channels = images.shape
    classes = 2                 # hard coded for now
    contour_type = {'inner': 'i', 'outer': 'o'}[args.classes]

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

    for path in patient_dirs:
        ret = load_patient_images(path, args.normalize)
        images, patient_number, frame_indices, rotated = ret

        predictions = []
        for image in images:
            mask_pred = m.predict(image[None,:,:,:]) # feed one at a time
            predictions.append((image[:,:,0], mask_pred[0,:,:,1]))

        for (image, mask), frame_index in zip(predictions, frame_indices):
            filename = "P{:02d}-{:04d}-{}contour-auto.txt".format(
                patient_number, frame_index, contour_type)
            outpath = os.path.join(args.outdir, filename)
            print(filename)

            contour = get_contours(mask)
            if rotated:
                height, width = image.shape
                x, y = contour.T
                x, y = height - y, x
                contour = np.vstack((x, y)).T

            np.savetxt(outpath, contour, fmt='%i', delimiter=' ')

            if args.checkpoint:
                filename = "P{:02d}-{:04d}-{}contour-auto.png".format(
                    patient_number, frame_index, contour_type)
                outpath = os.path.join(args.outdir, filename)
                save_image(outpath, image, np.round(mask))

if __name__ == '__main__':
    main()
