#!/usr/bin/env python

from __future__ import division, print_function

import os, glob, re
import dicom
import numpy as np
from PIL import Image, ImageDraw


def maybe_rotate(image):
    return np.rot90(image) if image.shape == (216, 256) else image

class PatientData(object):
    """Data directory structure (for patient 01):
    directory/
      P01dicom.txt
      P01dicom/
        P01-0000.dcm
        P01-0001.dcm
        ...
      P01contours-manual/
        P01-0080-icontour-manual.txt
        P01-0120-ocontour-manual.txt
        ...
    """
    def __init__(self, directory):
        self.directory = os.path.normpath(directory)

        # get patient index from contour listing file
        glob_search = os.path.join(directory, "P*list.txt")
        self.contour_list_file = glob.glob(glob_search)[0]
        match = re.search("P(..)list.txt", self.contour_list_file)
        self.index = int(match.group(1))

        # load all data into memory
        self.load_images()
        self.load_masks()

    @property
    def images(self):
        return [self.all_images[i] for i in self.labeled]

    @property
    def dicoms(self):
        return [self.all_dicoms[i] for i in self.labeled]

    @property
    def dicom_path(self):
        return os.path.join(self.directory, "P{:02d}dicom".format(self.index))

    def load_images(self):
        glob_search = os.path.join(self.dicom_path, "*.dcm")
        dicom_files = glob.glob(glob_search)
        self.all_images = []
        self.all_dicoms = []
        for dicom_file in dicom_files:
            plan = dicom.read_file(dicom_file)
            image = plan.pixel_array.astype(float)
            image *= 255/image.max()
            self.all_images.append(np.asarray(image, dtype='uint8'))
            self.all_dicoms.append(plan)
        self.image_height, self.image_width = plan.pixel_array.shape

    def load_masks(self):
        with open(self.contour_list_file, 'r') as f:
            files = [line.strip() for line in f.readlines()]

        inner_files = [path.replace("\\", "/") for path in files[0::2]]
        outer_files = [path.replace("\\", "/") for path in files[1::2]]

        self.labeled = []
        self.endocardium_masks = []
        self.epicardium_masks = []
        for inner_file, outer_file in zip(inner_files, outer_files):
            match = re.search("patient../(.*)", inner_file)
            inner_path = os.path.join(self.directory, match.group(1))
            inner_x, inner_y = np.loadtxt(inner_path).T

            match = re.search("patient../(.*)", outer_file)
            outer_path = os.path.join(self.directory, match.group(1))
            outer_x, outer_y = np.loadtxt(outer_path).T

            match = re.search("P..-(....)-.contour", inner_file)
            frame_number = int(match.group(1))
            self.labeled.append(frame_number)

            BW_8BIT = 'L'
            polygon = list(zip(inner_x, inner_y))
            image_dims = (self.image_width, self.image_height)
            img = Image.new(BW_8BIT, image_dims, color=0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            mask = np.array(img)
            self.endocardium_masks.append(mask)

            polygon = list(zip(outer_x, outer_y))
            image_dims = (self.image_width, self.image_height)
            img = Image.new(BW_8BIT, image_dims, color=0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            mask = np.array(img)
            self.epicardium_masks.append(mask)
            
    def write_video(self, outfile, FPS = 24):
        # Not converted to class interface yet -- NOT WORKING YET
        import cv2
        image_dims = (self.image_width, self.image_height)
        video = cv2.VideoWriter(outfile, -1, FPS, image_dims)
        for dicom_file in dicom_files:
            plan = dicom.read_file(dicom_file)
            frame = plan.pixel_array.astype(float)
            scale = max(frame.flat)
            image = np.asarray(255*frame/scale, dtype='uint8')
            video.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        video.release()

