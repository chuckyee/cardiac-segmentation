from __future__ import division, print_function

from keras.layers import Input, Conv2D, Concatenate
from keras.models import Model


class DilatedDenseNet(object):
    def __init__(self, height, width, channels, classes, features=64, depth=4,
                 temperature=1.0, padding='valid', batchnorm=False,
                 dropout=0.0):
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes
        self.features = features
        self.depth = depth
        self.temperature = temperature
        self.padding = padding

        inputs, outputs = self.build_model()
        
        super(DilatedDenseNet, self).__init__(inputs=inputs, outputs=outputs)

    def build_model(self):
        x0 = Input(shape=(self.height, self.width, self.channels))
        inputs = x0

        conv_args = {
            'padding': self.padding,
            'activation': 'relu',
        }
        filters = self.features
        kernel_size = (3,3)

        x1 = Conv2D(filters, kernel_size, dilation_rate=1, **conv_args)(x0)

        x2 = Concatenate()([x0, x1])
        x2 = Conv2D(filters, kernel_size, dilation_rate=2, **conv_args)(x2)
        
        x3 = Concatenate()([x0, x1, x2])
        x3 = Conv2D(filters, kernel_size, dilation_rate=4, **conv_args)(x3)

        x4 = Concatenate()([x0, x1, x2, x3])
        x4 = Conv2D(filters, kernel_size, dilation_rate=8, **conv_args)(x4)

        x5 = Concatenate()([x0, x1, x2, x3, x4])
        x5 = Conv2D(filters, kernel_size, dilation_rate=16, **conv_args)(x5)

        x6 = Concatenate()([x0, x1, x2, x3, x4, x5])
        x6 = Conv2D(filters, kernel_size, dilation_rate=32, **conv_args)(x6)

        x7 = Concatenate()([x0, x1, x2, x3, x4, x5, x6])
        x7 = Conv2D(filters, kernel_size, dilation_rate=64, **conv_args)(x7)

        x8 = Concatenate()([x0, x1, x2, x3, x4, x5, x6, x7])
        x8 = Conv2D(filters, kernel_size, dilation_rate=128, **conv_args)(x8)

        probabilities = Conv2D(self.classes, kernel_size=(1,1), activation='softmax')(x8)

        return inputs, probabilities
