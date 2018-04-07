

"""

    this file will implement the gradients of all operations defined by our Operations classes
"""


from __future__ import print_function
from __future__ import division

import numpy as np

# this global variable will store all of our gradients
all_gradients = {}


class Gradients(object):

    """
        this will define a basic interface for our a wrapper for all of the Gradients
    """

    def __init__(self, operation):
        # will receive the function whose gradient we want to save
        self.op = eval(operation)

        pass


    def __call__(self, gradient_function):
        # this simply assigns a gradient to the function
        all_gradients[self.op] = gradient_function

        pass

"""
    Now we shall start to define the actual gradient functions
    ::upstream_gradient will be coming from the output side of the operator
"""

@Gradients("add")
def add_gradient(upstream_gradient):
    # just pass them as it is!
    return  [upstream_gradient, upstream_gradient]


@Gradients("dot")
def dot_gradient(operation, upstream_gradient):
    # the gradients are swapped!
    dX = np.dot(operation.prev_nodes[0].matrix, upstream_gradient)
    dW = np.dot(operation.prev_nodes[1].matrix, upstream_gradient)

    return dX, dW















