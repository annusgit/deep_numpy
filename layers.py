

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



class Sigmoid(Layer):

    """
        our sigmoid operation; will be treated as another layer; yet to be defined
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(Sigmoid, self).__init__(input_nodes)

        # we might need their shapes at some point
        # print(input_nodes[0].shape, input_nodes[1].shape)
        self.shape = input_nodes[0].shape
        pass


    def compute(self):

        # A and B are two actual matrices that we want to add
        x = self.prev_nodes[0].output
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = 1 / (1 + np.exp(x))

        return self.output



class Relu(Layer):

    """
        our Relu operation; will be treated as another layer; yet to be defined
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(Relu, self).__init__(input_nodes)

        # we might need their shapes at some point
        # print(input_nodes[0].shape, input_nodes[1].shape)
        self.shape = input_nodes[0].shape
        pass


    def compute(self):

        # A and B are two actual matrices that we want to add
        x = self.prev_nodes[0].output
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = x * (x > 0)

        return self.output


class Softmax(Layer):

    """
        our softmax squashing operation to convert numbers in real probabilities
    """

    def __init__(self, *input_nodes):

        super(Softmax, self).__init__(input_nodes)

        # we might need their shapes at some point
        self.shape = input_nodes[0].shape
        pass


    def compute(self):

        # input_matrix is the computation of the last layer before softmax
        # print(len(self.prev_nodes))
        input_matrix = self.prev_nodes[0].output

        # exp sum
        exps = np.exp(input_matrix)
        # print(exps.shape)
        self.output = exps / np.sum(exps, axis=1)[:,None]
        # print(self.output.shape)
        return self.output













