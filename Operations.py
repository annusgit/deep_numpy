

"""
    this file contains the actual implementations of some real operations such as multiplication, addition etc.
    for matrices
"""


from __future__ import division
from graph_and_ops import*
import numpy as np



class add(Operation):

    """
        our add operation; will be treated as another operation
    """

    def __init__(self, *input_nodes):

        super(add, self).__init__(input_nodes)
        pass


    def compute(self):

        # A and B are two actual matrices that we want to add
        input_matrices = [node.output for node in self.prev_nodes]
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.add(input_matrices[0], input_matrices[1])

        return self.output



class dot(Operation):

    """
        our dot operation; will be treated as another operation; yet to be defined
    """

    def __init__(self, *input_nodes):

        super(dot, self).__init__(input_nodes)
        pass


    def compute(self):

        # A and B are two actual matrices that we want to add
        input_matrices = [node.output for node in self.prev_nodes]
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.dot(input_matrices[0], input_matrices[1])

        return self.output



class softmax_classifier(Operation):

    """
        our softmax squashing operation to convert numbers in real probabilities
    """

    def __init__(self, *input_nodes):

        super(softmax_classifier, self).__init__(input_nodes)
        pass


    def compute(self):

        # input_matrix is the computation of the last layer before softmax
        # print(len(self.prev_nodes))
        input_matrix = self.prev_nodes[0].output

        # exp sum
        exps = np.exp(input_matrix)
        self.output = exps / np.sum(exps, axis=1)[:,None]

        return self.output
















