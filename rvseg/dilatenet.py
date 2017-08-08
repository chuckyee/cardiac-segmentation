from __future__ import division, print_function

from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K



class DilatedUNet(object):
    def __init__(self, height, width, channels,
                 features=64, depth=4, classes=2, temperature=1.0,
                 padding='valid', batchnorm=False, dropout=0.0):
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes

    def generate_model(self):
        x = Input(shape=(self.height, self.width, self.channels))
        inputs = x

        x = Conv2D(32, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
        x = Conv2D(32, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x1 = x

        x = Conv2D(64, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x2 = x

        x = Conv2D(128, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
        x = Conv2D(128, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x3 = x

        x = Conv2D(256, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
        x = Conv2D(256, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)

        x = Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2))(x)
        x = Concatenate()([x, x3])
        x = Conv2D(128, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
        x = Conv2D(128, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
    
        x = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2))(x)
        x = Concatenate()([x, x2])
        x = Conv2D(64, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)

        x = Conv2DTranspose(32, kernel_size=(2,2), strides=(2,2))(x)
        x = Concatenate()([x, x1])
        x = Conv2D(32, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
        x = Conv2D(32, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)

        probabilities = Conv2D(self.classes, kernel_size=(1,1), activation='softmax')(x)

        return Model(inputs=inputs, outputs=probabilities)
