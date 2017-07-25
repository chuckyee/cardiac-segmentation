from __future__ import division, print_function

import unittest

import os
import dicom
import numpy as np
import patient

class TestPatientData(unittest.TestCase):
    def setUp(self):
        # test data has frames 0,1,..21, with the 20th frame labeled
        self.directory = "../test-assets/patient09/"
        self.dicom_path = self.directory + "P09dicom"

    def test_patient_data(self):
        p = patient.PatientData(self.directory)
        self.assertEqual(p.index, 9)
        self.assertEqual(p.dicom_path, self.dicom_path)
        self.assertEqual(len(p.images), 1)
        self.assertEqual(len(p.dicoms), 1)
        self.assertEqual(len(p.all_images), 22)
        self.assertEqual(len(p.all_dicoms), 22)
        self.assertEqual(p.image_width, 256)
        self.assertEqual(p.image_height, 216)
        self.assertEqual(p.rotated, True)
        self.assertEqual(p.labeled, [20])
        self.assertEqual(len(p.endocardium_masks), 1)
        self.assertEqual(len(p.epicardium_masks), 1)

        # check dicom MRI image
        plan = dicom.read_file(self.directory + "P09dicom/P09-0000.dcm")
        np.testing.assert_array_equal(p.all_dicoms[0].pixel_array,
                                      plan.pixel_array)

        # check endo- and epicardium masks
        endo_mask = np.loadtxt(self.directory + "endocardium-p09-0020.mask")
        np.testing.assert_array_equal(p.endocardium_masks[0], endo_mask)
        epi_mask = np.loadtxt(self.directory + "epicardium-p09-0020.mask")
        np.testing.assert_array_equal(p.epicardium_masks[0], epi_mask)

    def test_write_video(self):
        p = patient.PatientData(self.directory)
        outfile = "test_write_video.mp4"
        path = os.path.join(os.getcwd(), outfile)
        p.write_video(path)
        # simply check file exists and isn't empty
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)
        os.remove(path)
