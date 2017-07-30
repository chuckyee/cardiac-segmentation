from __future__ import division, print_function

import unittest

from keras import backend as K

from rvsc import loss

class TestModel(unittest.TestCase):
    def test_sorensen_dice(self):
        y_true = K.constant([[[0, 1], [1, 0]],
                             [[1, 0], [1, 0]]])
        y_pred = K.constant([[[.4, .6], [.1, .9]],
                             [[.5, .5], [.0, 1.]]])
        dice_coefs = loss.soft_sorensen_dice(y_true, y_pred, axis=[0, 1])
        dice_coefs = K.eval(dice_coefs)
        expected_dice_coefs = [0.44, 0.44]
        for x,y in zip(dice_coefs, expected_dice_coefs):
            self.assertAlmostEqual(x, y)

        dice_coefs = loss.hard_sorensen_dice(y_true, y_pred, axis=[0, 1])
        expected_dice_coefs = []
