from __future__ import division, print_function

import os
import argparse
import configparser
import logging

definitions = [
    # model               type   default help
    ('model',            (str,   'unet', "Model: unet, dilated-unet, dilated-densenet")),
    ('features',         (int,   64,     "Number of features maps after first convolutional layer.")),
    ('depth',            (int,   4,      "Number of downsampled convolutional blocks.")),
    ('temperature',      (float, 1.0,    "Temperature of final softmax layer in model.")),
    ('padding',          (str,   'same', "Padding in convolutional layers. Either `same' or `valid'.")),
    ('dropout',          (float, 0.0,    "Rate for dropout of activation units.")),
    ('classes',          (str,   'inner', "One of `inner' (endocardium), `outer' (epicardium), or `both'.")),
    ('batchnorm',        {'default': False, 'action': 'store_true', 
                          'help': "Apply batch normalization before nonlinearities."}),

    # loss
    ('loss',             (str,   'pixel', "Loss function: `pixel' for pixel-wise cross entropy, `dice' for dice coefficient.")),
    ('loss-weights',     {'type': float, 'nargs': '+', 'default': [0.1, 0.9],
                          'help': "When using dice or jaccard loss, how much to weight each output class."}),

    # training
    ('epochs',           (int,   20,     "Number of epochs to train.")),
    ('batch-size',       (int,   32,     "Mini-batch size for training.")),
    ('validation-split', (float, 0.2,    "Percentage of training data to hold out for validation.")),
    ('optimizer',        (str,   'adam', "Optimizer: sgd, rmsprop, adagrad, adadelta, adam, adamax, or nadam.")),
    ('learning-rate',    (float, None,   "Optimizer learning rate.")),
    ('momentum',         (float, None,   "Momentum for SGD optimizer.")),
    ('decay',            (float, None,   "Learning rate decay (not applicable for nadam).")),
    ('shuffle_train_val', {'default': False, 'action': 'store_true',
                           'help': "Shuffle images before splitting into train vs. val."}),
    ('shuffle',          {'default': False, 'action': 'store_true',
                          'help': "Shuffle images before each training epoch."}),
    ('seed',             (int,   None,   "Seed for numpy RandomState")),

    # files
    ('datadir',          (str,   '.',    "Directory containing patientXX/ directories.")),
    ('outdir',           (str,   '.',    "Directory to write output data.")),
    ('outfile',          (str,   'weights-final.hdf5', "File to write final model weights.")),
    ('load-weights',     (str,   '',     "Load model weights from specified file to initialize training.")),
    ('checkpoint',       {'default': False, 'action': 'store_true',
                          'help': "Write model weights after each epoch if validation accuracy improves."}),

    # augmentation
    ('augment-training', {'default': False, 'action': 'store_true',
                          'help': "Whether to apply image augmentation to training set."}),
    ('augment-validation', {'default': False, 'action': 'store_true',
                            'help': "Whether to apply image augmentation to validation set."}),
    ('rotation-range',     (float, 180,    "Rotation range (0-180 degrees)")),
    ('width-shift-range',  (float, 0.1,    "Width shift range, as a float fraction of the width")),
    ('height-shift-range', (float, 0.1,    "Height shift range, as a float fraction of the height")),
    ('shear-range',        (float, 0.1,    "Shear intensity (in radians)")),
    ('zoom-range',         (float, 0.05,   "Amount of zoom. If a scalar z, zoom in [1-z, 1+z]. Can also pass a pair of floats as the zoom range.")),
    ('fill-mode',          (str,   'nearest', "Points outside boundaries are filled according to mode: constant, nearest, reflect, or wrap")),
    ('alpha',              (float, 500,    "Random elastic distortion: magnitude of distortion")),
    ('sigma',              (float, 20,     "Random elastic distortion: length scale")),
    ('normalize', {'default': False, 'action': 'store_true',
                   'help': "Subtract mean and divide by std dev from each image."}),
]

noninitialized = {
    'learning_rate': 'getfloat',
    'momentum': 'getfloat',
    'decay': 'getfloat',
    'seed': 'getint',
}

def update_from_configfile(args, default, config, section, key):
    # Point of this function is to update the args Namespace.
    value = config.get(section, key)
    if value == '' or value is None:
        return

    # Command-line arguments override config file values
    if getattr(args, key) != default:
        return

    # Config files always store values as strings -- get correct type
    if isinstance(default, bool):
        value = config.getboolean(section, key)
    elif isinstance(default, int):
        value = config.getint(section, key)
    elif isinstance(default, float):
        value = config.getfloat(section, key)
    elif isinstance(default, str):
        value = config.get(section, key)
    elif isinstance(default, list):
        # special case (HACK): loss-weights is list of floats
        string = config.get(section, key)
        value = [float(x) for x in string.split()]
    elif default is None:
        # values which aren't initialized
        getter = getattr(config, noninitialized[key])
        value = getter(section, key)
    setattr(args, key, value)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train U-Net to segment right ventricles from cardiac "
        "MRI images.")

    for argname, kwargs in definitions:
        d = kwargs
        if isinstance(kwargs, tuple):
            d = dict(zip(['type', 'default', 'help'], kwargs))
        parser.add_argument('--' + argname, **d)

    # allow user to input configuration file
    parser.add_argument(
        'configfile', nargs='?', type=str, help="Load options from config "
        "file (command line arguments take precedence).")

    args = parser.parse_args()

    if args.configfile:
        logging.info("Loading options from config file: {}".format(args.configfile))
        config = configparser.ConfigParser(
            inline_comment_prefixes=['#', ';'], allow_no_value=True)
        config.read(args.configfile)
        for section in config:
            for key in config[section]:
                if key not in args:
                    raise Exception("Unknown option {} in config file.".format(key))
                update_from_configfile(args, parser.get_default(key),
                                       config, section, key)

    for k,v in vars(args).items():
        logging.info("{:20s} = {}".format(k, v))

    return args
