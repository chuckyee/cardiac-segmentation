from __future__ import print_function, division

import unittest
import densenet

class TestDenseNet(unittest.TestCase):
    def test_densenet(self):
        model = densenet.DenseNet(image_channels=1, num_init_features=12,
                                  growth_rate=12, layers=3)
        print(model)
        print(dir(model))
        print(model.initial_conv)
        print(model.features)
        print(dir(model.initial_conv))

    def test_denselayer(self):
        layer = densenet._DenseLayer(in_features=24, growth_rate=12)
        print(layer)
    
