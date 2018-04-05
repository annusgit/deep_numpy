

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








