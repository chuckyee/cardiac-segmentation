from __future__ import division, print_function

from keras.layers import Input, Conv2D, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.models import Model


def dilated_densenet(height, width, channels, classes, features=12, depth=4,
                     temperature=1.0, padding='same', batchnorm=False,
                     dropout=0.0):
    x = Input(shape=(height, width, channels))
    inputs = x

    # initial convolution
    x = Conv2D(features, kernel_size=(5,5), padding=padding)(x)

    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3,3)
    for n in range(depth):
        maps.append(x)
        x = Concatenate()(maps)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate,
                   padding=padding)(x)
        dilation_rate *= 2

    probabilities = Conv2D(classes, kernel_size=(1,1), activation='softmax')(x)

    return Model(inputs=inputs, outputs=probabilities)


def dilated_densenet2(height, width, channels, classes, features=12, depth=4,
                      temperature=1.0, padding='same', batchnorm=False,
                      dropout=0.0):
    x = Input(shape=(height, width, channels))
    inputs = x

    # initial convolution
    x = Conv2D(features, kernel_size=(5,5), padding=padding)(x)

    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3,3)
    for n in range(depth):
        maps.append(x)
        x = Concatenate()(maps)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(4*features, kernel_size=1)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate,
                   padding=padding)(x)
        dilation_rate *= 2

    probabilities = Conv2D(classes, kernel_size=(1,1), activation='softmax')(x)

    return Model(inputs=inputs, outputs=probabilities)


def dilated_densenet3(height, width, channels, classes, features=12, depth=4,
                      temperature=1.0, padding='same', batchnorm=False,
                      dropout=0.0):
    x = Input(shape=(height, width, channels))
    inputs = x

    # initial convolution
    x = Conv2D(features, kernel_size=(5,5), padding=padding)(x)

    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3,3)
    for n in range(depth):
        maps.append(x)
        x = Concatenate()(maps)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate,
                   padding=padding)(x)
        dilation_rate *= 2

    # Additional 2 layers to help generate segmentation mask
    x = Conv2D(features, kernel_size=(3,3), activation='relu', padding=padding)(x)
    x = Conv2D(features, kernel_size=(3,3), activation='relu', padding=padding)(x)

    probabilities = Conv2D(classes, kernel_size=(1,1), activation='softmax')(x)

    return Model(inputs=inputs, outputs=probabilities)
