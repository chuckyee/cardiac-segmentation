from __future__ import division, print_function

import unittest

import dataset

class TestDataset(unittest.TestCase):
    def test_keras_generator(self):
        self._test_generator(dataset.create_generators_keras)

    def test_keras_no_validation(self):
        self._test_no_validation(dataset.create_generators_keras)

    def test_generator(self):
        self._test_generator(dataset.create_generators)

    def test_no_validation(self):
        self._test_no_validation(dataset.create_generators)

    def _test_generator(self, create_generators):
        data_dir = "../test-assets/"
        batch_size = 2
        validation_split = 0.5
        # With a total of 3 training images, this split will create 1
        # training image and 2 validation images

        (train_generator, train_steps_per_epoch,
         val_generator, val_steps_per_epoch) = create_generators(
             data_dir, batch_size, validation_split)

        self.assertEqual(train_steps_per_epoch, 1)
        self.assertEqual(val_steps_per_epoch, 1)

        images, masks = next(train_generator)
        self.assertEqual(images.shape, (1, 216, 256, 1))
        self.assertEqual(masks.shape, (1, 216, 256, 2))

        images, masks = next(val_generator)
        self.assertEqual(images.shape, (2, 216, 256, 1))
        self.assertEqual(masks.shape, (2, 216, 256, 2))

    def _test_no_validation(self, create_generators):
        data_dir = "../test-assets/"
        batch_size = 2
        validation_split = 0.0

        (train_generator, train_steps_per_epoch,
         val_generator, val_steps_per_epoch) = create_generators(
             data_dir, batch_size, validation_split)

        self.assertEqual(train_steps_per_epoch, 2)
        self.assertEqual(val_steps_per_epoch, 0)

        # first 2 train images
        images, masks = next(train_generator)
        self.assertEqual(images.shape, (2, 216, 256, 1))
        self.assertEqual(masks.shape, (2, 216, 256, 2))

        # last train image (for total of 3)
        images, masks = next(train_generator)
        self.assertEqual(images.shape, (1, 216, 256, 1))
        self.assertEqual(masks.shape, (1, 216, 256, 2))

        # first 2 train images again
        images, masks = next(train_generator)
        self.assertEqual(images.shape, (2, 216, 256, 1))
        self.assertEqual(masks.shape, (2, 216, 256, 2))

        # validation generator should be nothing
        self.assertEqual(val_generator, None)
