from __future__ import division, print_function

from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K


def downsampling_block(input_tensor, filters, padding='valid',
                       batchnorm=False, dropout=0.0):
    _, height, width, _ = K.int_shape(input_tensor)
    assert height % 2 == 0
    assert width % 2 == 0

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return MaxPooling2D(pool_size=(2,2))(x), x

def upsampling_block(input_tensor, skip_tensor, filters, padding='valid',
                     batchnorm=False, dropout=0.0):
    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2))(input_tensor)

    # compute amount of cropping needed for skip_tensor
    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:
        cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    x = Concatenate()([x, y])

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return x

def unet(height, width, channels, classes, features=64, depth=4,
         temperature=1.0, padding='valid', batchnorm=False, dropout=0.0):
    """Generate U-Net model introduced in
      "U-Net: Convolutional Networks for Biomedical Image Segmentation"
      O. Ronneberger, P. Fischer, T. Brox (2015)
    Arbitrary number of input channels and output classes are supported.

    Arguments:
      height  - input image height (pixels)
      width   - input image width  (pixels)
      channels - input image features (1 for grayscale, 3 for RGB)
      classes - number of output classes (2 in paper)
      features - number of output features for first convolution (64 in paper)
          Number of features double after each down sampling block
      depth  - number of downsampling operations (4 in paper)
      padding - 'valid' (used in paper) or 'same'
      batchnorm - include batch normalization layers before activations
      dropout - fraction of units to dropout, 0 to keep all units

    Output:
      U-Net model expecting input shape (height, width, maps) and generates
      output with shape (output_height, output_width, classes). If padding is
      'same', then output_height = height and output_width = width.
    """
    x = Input(shape=(height, width, channels))
    inputs = x

    skips = []
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding,
                                   batchnorm, dropout)
        skips.append(x0)
        features *= 2

    x = Conv2D(filters=features, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters=features, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    for i in reversed(range(depth)):
        features //= 2
        x = upsampling_block(x, skips[i], features, padding,
                             batchnorm, dropout)

    x = Conv2D(filters=classes, kernel_size=(1,1))(x)

    logits = Lambda(lambda z: z/temperature)(x)
    probabilities = Activation('softmax')(logits)

    return Model(inputs=inputs, outputs=probabilities)
