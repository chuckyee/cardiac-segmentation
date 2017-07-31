from __future__ import division, print_function

import argparse

definitions = [
    (('-d', '--data_dir'),    (str,   '.',    "Directory containing patientXX/ directories")),
    (('-o', '--outfile'),     (str,   'weights-final.hdf5', "Directory to write output data")),
    (('--features',),         (int,   64,     "Number of features maps after first convolutional layer.")),
    (('--depth',),            (int,   4,      "Number of downsampled convolutional blocks.")),
    (('--temperature',),      (float, 1.0,    "Temperature of final softmax layer in model.")),
    (('--padding',),          (str,   'same', "Padding in convolutional layers. Either `same' or `valid'.")),
    (('--load_weights',),     (str,   None,   "Load model weights from specified file to initialize training.")),
    (('--learning_rate',),    (float, None,   "Optimizer learning rate.")),
    (('--momentum',),         (int,   None,   "Momentum for SGD optimizer.")),
    (('--decay',),            (int,   None,   "Learning rate decay (not applicable for nadam).")),
    (('--optimizer',),        (str,   'adam', "Optimizer: sgd, rmsprop, adagrad, adadelta, adam, adamax, or nadam.")),
    (('--loss',),             (str,   'pixel', "Loss function: `pixel' for pixel-wise cross entropy, `dice' for dice coefficient.")),
    (('-e', '--epochs'),      (int,   20,     "Number of epochs to train.")),
    (('-b', '--batch_size'),  (int,   32,     "Mini-batch size for training.")),
    (('--validation_split',), (float, 0.2,    "Percentage of training data to hold out for validation.")),
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train U-Net to segment right ventricles from cardiac MRI images.")
    for args, kwargs in definitions:
        d = dict(zip(['type', 'default', 'help'], kwargs))
        parser.add_argument(*args, **d)

    parser.add_argument('--checkpoint', default=False, action='store_true',
                        help="Write model weights after each epoch if validation accuracy improves.")
    parser.add_argument('--batch_norm', default=False, action='store_true',
                        help="Apply batch normalization before nonlinearities.")
    args = parser.parse_args()

    return args
