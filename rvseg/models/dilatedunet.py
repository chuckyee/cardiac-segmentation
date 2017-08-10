from __future__ import division, print_function

from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K


def dilated_unet(height, width, channels, classes, features=64, depth=4,
                 temperature=1.0, padding='valid', batchnorm=False,
                 dropout=0.0):
    x = Input(shape=(height, width, channels))
    inputs = x

    x = Conv2D(32, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
    x1 = x
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(64, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
    x2 = x
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
    x3 = x
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(256, kernel_size=(3,3), dilation_rate=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3,3), dilation_rate=2, padding='same', activation='relu')(x)
    # global context should exist at this level, in that each input pixel is mixed
    # However, high level features aren't globally incorporated into context
    x = Conv2D(256, kernel_size=(3,3), dilation_rate=4, padding='same', activation='relu')(x)
    # Adding additional dilated convolutional layers should globally incorporate context
    x = Conv2D(256, kernel_size=(3,3), dilation_rate=8, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3,3), dilation_rate=16, padding='same', activation='relu')(x)

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

    probabilities = Conv2D(classes, kernel_size=(1,1), activation='softmax')(x)

    return Model(inputs=inputs, outputs=probabilities)
