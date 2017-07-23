#!/usr/bin/env python

from __future__ import division, print_function

import os, glob
from PIL import Image
import patient


DATA_ROOT = "/Users/chuckyee/Developer/datasets/cardiac-mri/TrainingSet"

glob_search = os.path.join(DATA_ROOT, "patient*")
patient_dirs = glob.glob(glob_search)

print("Found patient directories:")
for patient_dir in patient_dirs:
    print(patient_dir)

train_dirs = patient_dirs[:13]
test_dirs = patient_dirs[13:]

x_train = "train/img/0"
y_train = "train/gt/0"
os.makedirs(x_train)
os.makedirs(y_train)

x_test = "test/img/0"
y_test = "test/gt/0"
os.makedirs(x_test)
os.makedirs(y_test)

file_format = "{:04d}.png"
BW_8BIT = 'L'

i = 0
for patient_dir in train_dirs:
    p = patient.PatientData(patient_dir)
    for image, endo in zip(p.images, p.endocardium_masks):
        x_outfile = os.path.join(x_train, file_format.format(i))
        Image.fromarray(image, BW_8BIT).save(x_outfile)
        y_outfile = os.path.join(y_train, file_format.format(i))
        Image.fromarray(endo, BW_8BIT).save(y_outfile)
        i += 1

i = 0
for patient_dir in test_dirs:
    p = patient.PatientData(patient_dir)
    for image, endo in zip(p.images, p.endocardium_masks):
        x_outfile = os.path.join(x_test, file_format.format(i))
        Image.fromarray(image, BW_8BIT).save(x_outfile)
        y_outfile = os.path.join(y_test, file_format.format(i))
        Image.fromarray(endo, BW_8BIT).save(y_outfile)
        i += 1
