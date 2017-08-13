from __future__ import division, print_function

import unittest
import numpy as np

from keras import backend as K

from rvseg import loss

class TestModel(unittest.TestCase):
    def sample_tensors(self):
        # have shapes (height, width, classes=2)
        y_true = K.constant([[[0, 1], [1, 0]],
                             [[1, 0], [1, 0]]])
        y_pred = K.constant([[[.4, .6], [.1, .9]],
                             [[.5, .5], [.0, 1.]]])
        return y_true, y_pred

    def test_sorensen_dice(self):
        y_true, y_pred = self.sample_tensors()
        dice_coefs = loss.soft_sorensen_dice(y_true, y_pred, axis=[0, 1])
        dice_coefs = K.eval(dice_coefs)
        # class 1: (2 * 0.6 + 1) / (1 + 3 + 1) = 0.44
        # class 2: (2 * 0.6 + 1) / (3 + 1 + 1) = 0.44
        expected_dice_coefs = [0.44, 0.44]
        for x,y in zip(dice_coefs, expected_dice_coefs):
            self.assertAlmostEqual(x, y)

        dice_coefs = loss.hard_sorensen_dice(y_true, y_pred, axis=[0, 1])
        dice_coefs = K.eval(dice_coefs)
        # class 1: (2 * 0 + 1) / (0 + 3 + 1) = 0.25
        # class 2: (2 * 1 + 1) / (3 + 1 + 1) = 0.6
        expected_dice_coefs = [0.25, 0.6]
        for x,y in zip(dice_coefs, expected_dice_coefs):
            self.assertAlmostEqual(x, y)

    def test_weighted_categorical_crossentropy(self):
        weights = [1, 9]
        y_true, y_pred = self.sample_tensors()

        lossfunc = loss.weighted_categorical_crossentropy(
            y_true, y_pred, weights)
        loss_val = K.eval(lossfunc)

        w = 2 * np.array(weights) / sum(weights)
        logs = -np.log(np.array([.1, .4, .5, .6, .9, 1e-8]))
        expected_loss_val = w[1]*logs[3]/4 + w[0]*(logs[0] + logs[2] + logs[5])/4

        self.assertAlmostEqual(np.mean(expected_loss_val), loss_val, places=5)

