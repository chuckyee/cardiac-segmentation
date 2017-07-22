#!/usr/bin/env python

from __future__ import division, print_function

import unittest

import os
import dicom
import patient

class TestPatientData(unittest.TestCase):
    def test_instantiation(self):
        directory = "../test-assets/patient09/"
        dicom_path = "../test-assets/patient09/P09dicom"
        p = patient.PatientData(directory)
        self.assertEqual(p.index, 9)
        self.assertEqual(p.dicom_path, dicom_path)
        self.assertEqual(len(p.images), 1)
        self.assertEqual(len(p.dicoms), 1)
        self.assertEqual(len(p.all_images), 22)
        self.assertEqual(len(p.all_dicoms), 22)
        self.assertEqual(p.labeled, [20])
        self.assertEqual(len(p.endocardium_masks), 1)
        self.assertEqual(len(p.epicardium_masks), 1)
