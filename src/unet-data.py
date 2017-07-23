#!/usr/bin/env python

from __future__ import division, print_function

import os, glob
import patient


DATA_ROOT = "/Users/chuckyee/Developer/datasets/cardiac-mri/TrainingSet"

glob_search = os.path.join(DATA_ROOT, "patient*")
patient_dirs = glob.glob(glob_search)

print("Found patient directories:")
for patient_dir in patient_dirs:
    print(patient_dir)

i = 0
for patient_dir in patient_dirs[:13]:
    p = patient.PatientData(patient_dir)
    for image, endo in zip(p.images, p.endocardium_masks):
        pass
