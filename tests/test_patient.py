from __future__ import division, print_function

import unittest

import os
import dicom
import numpy as np

from rvseg import patient

class TestPatientData(unittest.TestCase):
    def setUp(self):
        # test data has frames 0,1,..21, with the 20th frame labeled
        self.directory = "../test-assets/patient09/"
        self.dicom_path = self.directory + "P09dicom"

    def test_patient_data(self):
        p = patient.PatientData(self.directory)
        self.assertEqual(p.index, 9)
        self.assertEqual(p.dicom_path, self.dicom_path)
        self.assertEqual(len(p.images), 3)
        self.assertEqual(len(p.dicoms), 3)
        self.assertEqual(len(p.all_images), 24)
        self.assertEqual(len(p.all_dicoms), 24)
        self.assertEqual(p.image_width, 256)
        self.assertEqual(p.image_height, 216)
        self.assertEqual(p.rotated, True)
        self.assertEqual(p.labeled, [20, 22, 23])
        self.assertEqual(len(p.endocardium_masks), 3)
        self.assertEqual(len(p.epicardium_masks), 3)
        self.assertEqual(len(p.endocardium_contours), 3)
        self.assertEqual(len(p.epicardium_contours), 3)

        # check dicom MRI image
        plan = dicom.read_file(self.directory + "P09dicom/P09-0000.dcm")
        np.testing.assert_array_equal(p.all_dicoms[0].pixel_array,
                                      plan.pixel_array)
        self.assertEqual(p.images[0].dtype, 'uint16')

        # check endo- and epicardium masks
        endo_mask = np.loadtxt(self.directory + "endocardium-p09-0020.mask")
        np.testing.assert_array_equal(p.endocardium_masks[0], endo_mask)
        self.assertEqual(p.endocardium_masks[0].dtype, 'uint8')
        self.assertSetEqual(set(p.endocardium_masks[0].flatten()), set([0, 1]))

        epi_mask = np.loadtxt(self.directory + "epicardium-p09-0020.mask")
        np.testing.assert_array_equal(p.epicardium_masks[0], epi_mask)
        self.assertEqual(p.epicardium_masks[0].dtype, 'uint8')
        self.assertSetEqual(set(p.epicardium_masks[0].flatten()), set([0, 1]))

    @unittest.skip("Skipping video write: OpenCV is a pain to install in test env.")
    def test_write_video(self):
        p = patient.PatientData(self.directory)
        outfile = "test_write_video.mp4"
        path = os.path.join(os.getcwd(), outfile)
        p.write_video(path)
        # simply check file exists and isn't empty
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)
        os.remove(path)
