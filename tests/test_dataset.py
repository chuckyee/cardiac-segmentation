from __future__ import division, print_function

import unittest

import numpy as np
from rvseg import dataset

class TestDataset(unittest.TestCase):
    def test_generator(self):
        self._test_generator(mask='inner')
        self._test_generator(mask='outer')
        self._test_generator(mask='both')

    def test_no_validation(self):
        self._test_no_validation(mask='inner')
        self._test_no_validation(mask='outer')
        self._test_no_validation(mask='both')

    def _test_generator(self, mask):
        data_dir = "../test-assets/"
        batch_size = 2
        validation_split = 0.5
        # With a total of 3 training images, this split will create 1
        # training image and 2 validation images

        (train_generator, train_steps_per_epoch,
         val_generator, val_steps_per_epoch) = dataset.create_generators(
             data_dir, batch_size,
             validation_split=validation_split,
             mask=mask)

        self.assertEqual(train_steps_per_epoch, 1)
        self.assertEqual(val_steps_per_epoch, 1)

        classes = 3 if mask == 'both' else 2

        images, masks = next(train_generator)
        self.assertEqual(images.shape, (1, 216, 256, 1))
        self.assertEqual(masks.shape, (1, 216, 256, classes))

        images, masks = next(val_generator)
        self.assertEqual(images.shape, (2, 216, 256, 1))
        self.assertEqual(masks.shape, (2, 216, 256, classes))

    def _test_no_validation(self, mask):
        data_dir = "../test-assets/"
        batch_size = 2
        validation_split = 0.0

        (train_generator, train_steps_per_epoch,
         val_generator, val_steps_per_epoch) = dataset.create_generators(
             data_dir, batch_size,
             validation_split=validation_split,
             mask=mask)

        self.assertEqual(train_steps_per_epoch, 2)
        self.assertEqual(val_steps_per_epoch, 0)

        classes = 3 if mask == 'both' else 2

        # first 2 train images
        images, masks = next(train_generator)
        self.assertEqual(images.shape, (2, 216, 256, 1))
        self.assertEqual(masks.shape, (2, 216, 256, classes))

        # last train image (for total of 3)
        images, masks = next(train_generator)
        self.assertEqual(images.shape, (1, 216, 256, 1))
        self.assertEqual(masks.shape, (1, 216, 256, classes))

        # first 2 train images again
        images, masks = next(train_generator)
        self.assertEqual(images.shape, (2, 216, 256, 1))
        self.assertEqual(masks.shape, (2, 216, 256, classes))

        # validation generator should be nothing
        self.assertEqual(val_generator, None)


    def test_shuffle_train_val(self):
        # test shuffling of entire dataset prior to train-val split
        # (does not test shuffling within each epoch)
        data_dir = "../test-assets/"
        batch_size = 2
        validation_split = 0.5
        mask = "inner"
        classes = 2
        seed = 5               # random number seed

        # there should be 2 images in the validation set, and we'll check if
        # they always appear in the same order with a fixed seed
        image_list = []
        mask_list = []
        for i in range(10):
            _, _, val_generator, _ = dataset.create_generators(
                data_dir, batch_size, validation_split=validation_split,
                mask=mask, shuffle_train_val=True, shuffle=False, seed=seed,
                normalize_images=True)

            images, masks = next(val_generator)
            self.assertEqual(images.shape, (2, 216, 256, 1))
            self.assertEqual(masks.shape, (2, 216, 256, classes))

            # also check image normalization
            for image in images:
                self.assertAlmostEqual(np.mean(image), 0)
                self.assertAlmostEqual(np.std(image), 1, places=5)

            image_list.append(images[0])
            mask_list.append(masks[0])

        # first image/mask in each case should be the same
        image0 = image_list[0]
        for image in image_list[1:]:
            np.testing.assert_array_equal(image0, image)
        mask0 = mask_list[0]
        for mask in mask_list[1:]:
            np.testing.assert_array_equal(mask0, mask)

        # now test that things get shuffled if we don't specify a seed
        mask = "both"
        _, _, val_generator, _ = dataset.create_generators(
            data_dir, batch_size, validation_split=validation_split,
            mask=mask, shuffle_train_val=True, shuffle=False, seed=None,
            normalize_images=True)

        images, masks = next(val_generator)
        image0 = images[0]
        while 1:
            _, _, val_generator, _ = dataset.create_generators(
                data_dir, batch_size, validation_split=validation_split,
                mask=mask, shuffle_train_val=True, shuffle=True, seed=None,
                normalize_images=True)
            images, masks = next(val_generator)            
            try:
                np.testing.assert_array_equal(image0, images[0])
            except AssertionError:
                break           # break if arrays are differet (= success!)
