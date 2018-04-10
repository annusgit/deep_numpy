

"""

    this file will implement the gradients of all operations defined by our Operations classes
"""


from __future__ import print_function
from __future__ import division

import numpy as np

# this global variable will store all of our gradients against their functions
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
    
    Important:::
        Each gradient function will return a list of upstream_gradients, 
        as well as the adjustments for its own layer! 
"""

@Gradients("add")
def add_grad(upstream_gradient):
    # just pass them as it is!
    return  upstream_gradient


@Gradients("dot")
def dot_grad(operation, upstream_gradient):
    # the gradients are swapped!
    dX = np.dot(operation.prev_nodes[0].matrix, upstream_gradient)
    dW = np.dot(operation.prev_nodes[1].matrix, upstream_gradient)

    self.weight_grad = np.dot(upstream_gradient.transpose(), operation.prev_nodes[0].output)
    self.bias_grad = np.sum(upstream_gradient, axis=1)

    # this will be the upstream for the previous layer
    self.gradients = np.dot(upstream_gradient, operation.prev_nodes[1].output.transpose())

    return self.gradients

    return dX, dW


@Gradients("relu")
def relu_grad(relu_input, upstream_gradient):

    pass


@Gradients("Softmax_with_CrossEntropyLoss")
def softmax_crossent_grad(softmax_output, upstream_gradient):

    pass

















