#!/usr/bin/env python

from __future__ import division, print_function

import os, glob, re
import dicom
import numpy as np
from PIL import Image, ImageDraw


def maybe_rotate(image):
    # orient image in landscape
    height, width = image.shape
    return np.rot90(image) if width < height else image

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
        files = glob.glob(glob_search)
        if len(files) == 0:
            raise Exception("Couldn't find contour listing file in {}. "
                            "Wrong directory?".format(directory))
        self.contour_list_file = files[0]
        match = re.search("P(..)list.txt", self.contour_list_file)
        self.index = int(match.group(1))

        # load all data into memory
        self.load_images()

        # some patients do not have contour data, and that's ok
        try:
            self.load_masks()
        except FileNotFoundError:
            pass

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
        dicom_files = sorted(glob.glob(glob_search))
        self.all_images = []
        self.all_dicoms = []
        for dicom_file in dicom_files:
            plan = dicom.read_file(dicom_file)
            image = maybe_rotate(plan.pixel_array)
            self.all_images.append(image)
            self.all_dicoms.append(plan)
        self.image_height, self.image_width = image.shape
        self.rotated = (plan.pixel_array.shape != image.shape)

    def load_contour(self, filename):
        # strip out path head "patientXX/"
        match = re.search("patient../(.*)", filename)
        path = os.path.join(self.directory, match.group(1))
        x, y = np.loadtxt(path).T
        if self.rotated:
            x, y = y, self.image_height - x
        return x, y

    def contour_to_mask(self, x, y, norm=255):
        BW_8BIT = 'L'
        polygon = list(zip(x, y))
        image_dims = (self.image_width, self.image_height)
        img = Image.new(BW_8BIT, image_dims, color=0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        return norm * np.array(img, dtype='uint8')

    def load_masks(self):
        with open(self.contour_list_file, 'r') as f:
            files = [line.strip() for line in f.readlines()]

        inner_files = [path.replace("\\", "/") for path in files[0::2]]
        outer_files = [path.replace("\\", "/") for path in files[1::2]]

        # get list of frames which have contours
        self.labeled = []
        for inner_file in inner_files:
            match = re.search("P..-(....)-.contour", inner_file)
            frame_number = int(match.group(1))
            self.labeled.append(frame_number)

        self.endocardium_contours = []
        self.epicardium_contours = []
        self.endocardium_masks = []
        self.epicardium_masks = []
        for inner_file, outer_file in zip(inner_files, outer_files):
            inner_x, inner_y = self.load_contour(inner_file)
            self.endocardium_contours.append((inner_x, inner_y))
            outer_x, outer_y = self.load_contour(outer_file)
            self.epicardium_contours.append((outer_x, outer_y))

            inner_mask = self.contour_to_mask(inner_x, inner_y, norm=1)
            self.endocardium_masks.append(inner_mask)
            outer_mask = self.contour_to_mask(outer_x, outer_y, norm=1)
            self.epicardium_masks.append(outer_mask)
            
    def write_video(self, outfile, FPS=24):
        import cv2
        image_dims = (self.image_width, self.image_height)
        video = cv2.VideoWriter(outfile, -1, FPS, image_dims)
        for image in self.all_images:
            grayscale = np.asarray(image * (255 / image.max()), dtype='uint8')
            video.write(cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR))
        video.release()

