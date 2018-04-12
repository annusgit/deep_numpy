

"""

    this file will implement the gradients of all operations defined by our Operations classes
"""


from __future__ import print_function
from __future__ import division
from Associations import Association
import numpy as np

# this global variable will store all of our gradients against their functions
all_gradients = {}


"""
    Now we shall start to define the actual gradient functions
    ::upstream_gradient will be coming from the output side of the operator
    
    Important:::
        Each gradient function will return a list of upstream_gradients, 
        as well as the adjustments for its own layer! 
"""

@Association(operation='add', association='gradients')
def add_grad(upstream_gradient):
    # just pass them as it is!
    return  upstream_gradient


@Association(operation='dot', association='gradients')
def dot_grad(operation, upstream_gradient):
    # the gradients are swapped!
    dX = np.dot(operation.prev_nodes[0].matrix, upstream_gradient)
    dW = np.dot(operation.prev_nodes[1].matrix, upstream_gradient)

    weight_grad = np.dot(upstream_gradient.transpose(), operation.prev_nodes[0].output)
    bias_grad = np.sum(upstream_gradient, axis=1)

    # this will be the upstream for the previous layer
    gradients = np.dot(upstream_gradient, operation.prev_nodes[1].output.transpose())

    return dX, dW











