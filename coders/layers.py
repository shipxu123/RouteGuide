##
# @file layers.py
# @author Keren Zhu
# @date Feb 2019
# @brief Layers
#

import tensorflow as tf
import numpy as np

class Convolution2D(object):
    def __init__(self,
                kernel_shape,
                kernel=None,
                bias=None,
                strides=(1,1,1,1),
                padding='SAME',
                activation=None,
                scope=''):

        self.kernel_shape = kernel_shape
        self.kernel = kernel
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope

    def build(self, input_tensor):
        if (self.kernel):
            assert self.kernel.get_shape() == self.kernel_shape

