from __future__ import division, print_function

import unittest

from keras.layers import Input
from keras import backend as K
import model

class TestModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_downsampling(self):
        inputs = Input(shape=(28, 28, 1))
        filters = 16
        padding = 'valid'
        x, y = model.downsampling_block(inputs, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 12, 12, filters))
        self.assertTupleEqual(K.int_shape(y), (None, 24, 24, filters))

        padding = 'same'
        x, y = model.downsampling_block(inputs, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 14, 14, filters))
        self.assertTupleEqual(K.int_shape(y), (None, 28, 28, filters))

    def test_downsampling_error(self):
        inputs = Input(shape=(29, 29, 1))
        filters = 16
        with self.assertRaises(AssertionError):
            model.downsampling_block(inputs, filters, padding='valid')
        with self.assertRaises(AssertionError):
            model.downsampling_block(inputs, filters, padding='same')

    def test_upsampling(self):
        inputs = Input(shape=(14, 14, 32))
        skip = Input(shape=(28, 28, 16))
        filters = 16
        padding = 'valid'
        x = model.upsampling_block(inputs, skip, filters, padding)
        self.assertTuple(K.int_shape(x), ())
