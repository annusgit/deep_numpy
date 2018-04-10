
"""

    this file will implement the associations of all operations with their gradients and adjustments
"""


from __future__ import print_function
from __future__ import division

import numpy as np

# these global variables will store all of our gradients and adjustments against their functions
all_gradients = {}
all_adjustments = {}


class Association(object):

    """
        this will define a basic interface for our a wrapper for all of the Gradients
    """

    def __init__(self, operation, association='gradient'): # the association will be either 'gradient' or 'adjustment'
        # will receive the function whose gradient we want to save
        self.op = eval(operation)
        self.association = association
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


