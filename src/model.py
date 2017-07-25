from __future__ import division, print_function


from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import optimizers

import glob
import patient


def downsampling_block(input_tensor, filters, padding='valid'):
    x = Conv2D(filters=filters, kernel_size=(3,3), padding=padding, activation='relu')(input_tensor)
    x = Conv2D(filters=filters, kernel_size=(3,3), padding=padding, activation='relu')(x)
    return MaxPooling2D(pool_size=(2,2))(x), x

def upsampling_block(input_tensor, skip_tensor, filters, padding='valid'):
    x = UpSampling2D(size=(2,2))(input_tensor)
    # compute amount of cropping needed for skip_tensor
    if padding == 'valid':
        _, x_height, x_width, _ = x.shape
        _, s_height, s_width, _ = skip_tensor.shape
        h_crop = (s_height - x_height)
        w_crop = (s_width - x_width)
        cropping=((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
        y = Cropping2D(cropping=cropping)(skip_tensor)
    else:
        y = skip_tensor
    x = Concatenate()([x, y])
    x = Conv2D(filters=filters, kernel_size=(3,3), padding=padding, activation='relu')(x)    
    return Conv2D(filters=filters, kernel_size=(3,3), padding=padding, activation='relu')(x)    

def unet(height, width, padding):
    x = Input(shape=(height, width))

    x, x1 = downsampling_block(x, 64,  padding)
    x, x2 = downsampling_block(x, 128, padding)
    x, x3 = downsampling_block(x, 256, padding)
    x, x4 = downsampling_block(x, 512, padding)

    x = Conv2D(filters=1024, kernel_size=(3,3), padding=padding, activation='relu')(x)
    x = Conv2D(filters=1024, kernel_size=(3,3), padding=padding, activation='relu')(x)

    x = upsampling_block(x, x4, 512, padding)
    x = upsampling_block(x, x3, 256, padding)
    x = upsampling_block(x, x2, 128, padding)
    x = upsampling_block(x, x1, 64,  padding)

    logits = Conv2D(filters=2, kernel_size=(1,1))(x)

    return Model(inputs=inputs, outputs=logis)

def main():
    patient_dirs = glob.glob("~/Developer/datasets/cardiac-mri/TrainingSet/patient*")
    images = []
    masks = []
    for patient_dir in patient_dirs:
        p = patient.PatientData(patient_dir)
        images.append(p.images)
        masks.append(p.endocardium_masks)

    height, width = 216, 256
    learning_rate = 0.01
    momentum = 0.99
    decay = 0.0
    epochs = 100
    validation_split = 0.2
    padding = 'same'

    height, width = images[0].shape
    model = unet(height, width, padding)
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(images, masks, epochs=epochs, validation_split=validation_split)
