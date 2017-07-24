#!/usr/bin/env python

from __future__ import division, print_function

import os, glob
import argparse
from PIL import Image
import patient


def generate_unet_data(patient_dirs, x_dir, y_dir, file_format="{:04d}"):
    BW_8BIT = 'L'
    i = 0
    for patient_dir in patient_dirs:
        p = patient.PatientData(patient_dir)
        for image, endo in zip(p.images, p.endocardium_masks):
            x_outfile = os.path.join(x_dir, file_format.format(i) + ".jpg")
            Image.fromarray(image, BW_8BIT).save(x_outfile)
            y_outfile = os.path.join(y_dir, file_format.format(i) + ".png")
            Image.fromarray(endo, BW_8BIT).save(y_outfile)
            i += 1

def main(args):
    glob_search = os.path.join(args.indir, "patient*")
    patient_dirs = glob.glob(glob_search)

    print("Found patient directories:")
    for patient_dir in patient_dirs:
        print(patient_dir)

    split_index = int((1-args.split/100) * len(patient_dirs))
    train_dirs = patient_dirs[:split_index]
    test_dirs = patient_dirs[split_index:]
    print("First {} patients used as training set, remaining as test.".format(split_index))

    x_train = os.path.join(args.outdir, "train/img/0")
    y_train = os.path.join(args.outdir, "train/gt/0")
    os.makedirs(x_train)
    os.makedirs(y_train)

    x_test = os.path.join(args.outdir, "test/img/0")
    y_test = os.path.join(args.outdir, "test/gt/0")
    os.makedirs(x_test)
    os.makedirs(y_test)

    generate_unet_data(train_dirs, x_train, y_train)
    generate_unet_data(test_dirs, x_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate data for U-Net from RV MRI dicom images.")
    parser.add_argument('indir', default='.', help="TrainingSet/ directory")
    parser.add_argument('-o', '--outdir', default='.', help="Directory to write output data")
    parser.add_argument('-s', '--split', default=20, type=int, help='Percentage of patients used for test set')
    args = parser.parse_args()
    main(args)
