from __future__ import division, print_function

import unittest

from keras.layers import Input
from keras import backend as K

from rvseg.models import convunet
from rvseg.models import unet

class TestModel(unittest.TestCase):
    def test_downsampling(self):
        inputs = Input(shape=(28, 28, 1))
        filters = 16
        padding = 'valid'
        x, y = convunet.downsampling_block(inputs, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 12, 12, filters))
        self.assertTupleEqual(K.int_shape(y), (None, 24, 24, filters))

        padding = 'same'
        x, y = convunet.downsampling_block(inputs, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 14, 14, filters))
        self.assertTupleEqual(K.int_shape(y), (None, 28, 28, filters))

    def test_downsampling_error(self):
        # downsampling should fail on odd-integer dimension images
        inputs = Input(shape=(29, 29, 1))
        filters = 16
        with self.assertRaises(AssertionError):
            convunet.downsampling_block(inputs, filters, padding='valid')
        with self.assertRaises(AssertionError):
            convunet.downsampling_block(inputs, filters, padding='same')

    def test_upsampling(self):
        # concatenation without cropping
        filters = 16
        inputs = Input(shape=(14, 14, 2*filters))
        skip = Input(shape=(28, 28, filters))
        padding = 'valid'
        x = convunet.upsampling_block(inputs, skip, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 24, 24, filters))

        # ((4,4), (4,4)) cropping
        filters = 15
        inputs = Input(shape=(10, 10, 2*filters))
        skip = Input(shape=(28, 28, filters))
        padding = 'valid'
        x = convunet.upsampling_block(inputs, skip, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 16, 16, filters))

        # odd-integer input size
        filters = 4
        inputs = Input(shape=(11, 11, 2*filters))
        skip = Input(shape=(28, 28, filters))
        padding = 'valid'
        x = convunet.upsampling_block(inputs, skip, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 18, 18, filters))

        # test odd-integer cropping
        filters = 5
        inputs = Input(shape=(11, 11, 2*filters))
        skip = Input(shape=(27, 27, filters))
        padding = 'valid'
        x = convunet.upsampling_block(inputs, skip, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 18, 18, filters))

        # test same padding
        filters = 5
        inputs = Input(shape=(11, 11, 2*filters))
        skip = Input(shape=(27, 27, filters))
        padding = 'same'
        x = convunet.upsampling_block(inputs, skip, filters, padding)
        self.assertTupleEqual(K.int_shape(x), (None, 22, 22, filters))

    def test_upsampling_error(self):
        filters = 2
        inputs = Input(shape=(11, 11, 2*filters))
        padding = 'valid'
        with self.assertRaises(AssertionError):
            skip = Input(shape=(21, 22, filters))
            x = convunet.upsampling_block(inputs, skip, filters, padding)
        with self.assertRaises(AssertionError):
            skip = Input(shape=(22, 21, filters))
            x = convunet.upsampling_block(inputs, skip, filters, padding)

    def test_unet(self):
        # classic u-net architecture from
        #   "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        #   O. Ronneberger, P. Fischer, T. Brox (2015)
        height, width, channels = 572, 572, 1
        features = 64
        depth = 4
        classes = 2
        temperature = 1.0
        padding = 'valid'
        m = unet(height, width, channels, classes, features, depth,
                 temperature, padding)
        self.assertEqual(len(m.layers), 56)

        # input/output dimensions
        self.assertTupleEqual(K.int_shape(m.input), (None, 572, 572, 1))
        self.assertTupleEqual(K.int_shape(m.output), (None, 388, 388, 2))

        # layers
        layer_output_dims = [
            (None, 572, 572, 1), # input
            (None, 570, 570, 64),
            (None, 570, 570, 64),
            (None, 568, 568, 64), # skip 1
            (None, 568, 568, 64),
            (None, 284, 284, 64), # max pool 2x2
            (None, 282, 282, 128),
            (None, 282, 282, 128),
            (None, 280, 280, 128), # skip 2
            (None, 280, 280, 128),
            (None, 140, 140, 128), # max pool 2x2
            (None, 138, 138, 256),
            (None, 138, 138, 256),
            (None, 136, 136, 256), # skip 3
            (None, 136, 136, 256),
            (None, 68, 68, 256), # max pool 2x2
            (None, 66, 66, 512),
            (None, 66, 66, 512),
            (None, 64, 64, 512), # skip 4
            (None, 64, 64, 512),
            (None, 32, 32, 512), # max pool 2x2
            (None, 30, 30, 1024),
            (None, 30, 30, 1024),
            (None, 28, 28, 1024),
            (None, 28, 28, 1024),
            (None, 56, 56, 512), # up-conv 2x2
            (None, 56, 56, 512), # cropping of skip 4
            (None, 56, 56, 1024), # concat
            (None, 54, 54, 512),
            (None, 54, 54, 512),
            (None, 52, 52, 512),
            (None, 52, 52, 512),
            (None, 104, 104, 256), # up-conv 2x2
            (None, 104, 104, 256), # cropping of skip 3
            (None, 104, 104, 512), # concat
            (None, 102, 102, 256),
            (None, 102, 102, 256),
            (None, 100, 100, 256),
            (None, 100, 100, 256),
            (None, 200, 200, 128), # up-conv 2x2
            (None, 200, 200, 128), # cropping of skip 2
            (None, 200, 200, 256), # concat
            (None, 198, 198, 128),
            (None, 198, 198, 128),
            (None, 196, 196, 128),
            (None, 196, 196, 128),
            (None, 392, 392, 64), # up-conv 2x2
            (None, 392, 392, 64), # cropping of skip 1
            (None, 392, 392, 128), # concat
            (None, 390, 390, 64),
            (None, 390, 390, 64),
            (None, 388, 388, 64),
            (None, 388, 388, 64),
            (None, 388, 388, 2), # output segmentation map
            (None, 388, 388, 2),
            (None, 388, 388, 2),
        ]
        for layer, shape in zip(m.layers, layer_output_dims):
            self.assertTupleEqual(layer.output_shape, shape)

    def check_layer_dims(self, model):
        # if we include only one of batch normalization or dropout,
        # then the shape of the network should be the same.
        layer_output_dims = [
            (None, 10, 10, 1), # input
            (None, 10, 10, 4), # conv2D
            (None, 10, 10, 4), # batchnorm | reLU
            (None, 10, 10, 4), # reLU      | dropout
            (None, 10, 10, 4), # conv2D
            (None, 10, 10, 4), # batchnorm | reLU
            (None, 10, 10, 4), # reLU      | dropout
            (None, 5, 5, 4),   # max pool 2x2
            (None, 5, 5, 8),   # conv2D
            (None, 5, 5, 8),   # batchnorm | reLU
            (None, 5, 5, 8),   # reLU      | dropout
            (None, 5, 5, 8),   # conv2D
            (None, 5, 5, 8),   # batchnorm | reLU
            (None, 5, 5, 8),   # reLU      | dropout
            (None, 10, 10, 4), # up-conv 2x2
            (None, 10, 10, 8), # concat
            (None, 10, 10, 4), # conv2D
            (None, 10, 10, 4), # batchnorm | reLU
            (None, 10, 10, 4), # reLU      | dropout
            (None, 10, 10, 4), # conv2D
            (None, 10, 10, 4), # batchnorm | reLU
            (None, 10, 10, 4), # reLU      | dropout
            (None, 10, 10, 2), # output segmentation map
            (None, 10, 10, 2), # (temperature)
            (None, 10, 10, 2), # softmax
        ]
        for layer, shape in zip(model.layers, layer_output_dims):
            self.assertTupleEqual(layer.output_shape, shape)

    def test_batchnorm(self):
        # only batch norm, no dropout
        height, width, channels = 10, 10, 1
        features = 4
        depth = 1
        classes = 2
        temperature = 1.0
        padding = 'same'
        batchnorm = True
        dropout = False
        m = unet(height, width, channels, classes, features, depth,
                 temperature, padding, batchnorm, dropout)
        self.assertEqual(len(m.layers), 25)

        # input/output dimensions
        self.assertTupleEqual(K.int_shape(m.input), (None, 10, 10, 1))
        self.assertTupleEqual(K.int_shape(m.output), (None, 10, 10, 2))

        self.check_layer_dims(m)

    def test_dropout(self):
        # only dropout, no batch norm
        height, width, channels = 10, 10, 1
        features = 4
        depth = 1
        classes = 2
        temperature = 1.0
        padding = 'same'
        batchnorm = False
        dropout = True
        m = unet(height, width, channels, classes, features, depth,
                 temperature, padding, batchnorm, dropout)
        self.assertEqual(len(m.layers), 25)

        # input/output dimensions
        self.assertTupleEqual(K.int_shape(m.input), (None, 10, 10, 1))
        self.assertTupleEqual(K.int_shape(m.output), (None, 10, 10, 2))

        self.check_layer_dims(m)
