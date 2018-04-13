

"""
    this section will define some full layers instead of operations
    all of the layers will be based on operations defined by Operations
"""

from __future__ import print_function
from __future__ import division

import numpy as np
from Operations import*


class fully_connected(Layer):
    """
        A simple dense layer
    """
    def __init__(self, features, units):

        super(fully_connected, self).__init__([features])

        # initialize its ops
        # print(features.shape)
        self.W = np.random.uniform(low=-0.1,high=0.1,size=(features.shape[1],units))
        self.bias = np.ones(shape=(features.shape[0], units))
        # print(self.W.shape, self.bias.shape)

        # this will be our connection to the network
        # this is the shape of the matrix that will come at the output!!!
        self.shape = (features.shape[0], units)

        # will be trainable
        self.is_trainable = True


    def compute(self):

        # just compute the dot and then add the bias
        input_matrix = self.prev_nodes[0].output
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.add(np.dot(input_matrix, self.W), self.bias)

        return self.output


    def back(self):

        # get the upstream gradient from the parent method
        super(fully_connected, self).back()

        # these will be required at weight update
        # print(self.W.shape)
        self.weight_grad = np.dot(self.prev_nodes[0].output.transpose(), self.upstream_grad)
        # self.bias_grad = np.sum(upstream_grad, axis=0)
        self.bias_grad = self.upstream_grad

        # this will be the upstream for the previous layer
        self.upstream_grad = np.dot(self.upstream_grad, self.W.transpose())


    def update(self, lr):

        """
            this method will be used to update our weights
        :return: None
        """
        self.W += -lr * self.weight_grad
        self.bias += -lr * self.bias_grad
        # print(self.W.shape, self.weight_grad.shape)
        # print(self.bias.shape, self.bias_grad.shape, np.sum(self.upstream, axis=0).shape)
        pass








