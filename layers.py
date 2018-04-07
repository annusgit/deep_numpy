

"""
    this section will define some full layers instead of operations
    all of the layers will be based on operations defined by Operations
"""

from __future__ import print_function
from __future__ import division

import numpy as np
from graph_and_ops import Layer
from Operations import*



class Dense(Layer):
    """
        A simple dense layer
    """
    def __init__(self, features, units):

        super(Dense, self).__init__([features])

        # initialize its ops
        # print(features.shape)
        self.W = np.random.uniform(low=-0.1,high=0.1,size=(features.shape[1],units))
        self.bias = np.random.uniform(low=-0.1,high=0.1,size=units)

        # this will be our connection to the network
        self.shape = (features.shape[1], units)


    def compute(self):

        # just compute the dot and then add the bias
        input_matrix = self.prev_nodes[0].output
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.add(np.dot(input_matrix, self.W), self.bias)

        return self.output

















