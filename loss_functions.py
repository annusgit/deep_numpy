

"""

    Here we shall implement some commonly used loss functions,
    contemporary losses may be defined using basic operations from Operations
"""


from __future__ import print_function
from __future__ import division

import numpy as np
from graph_and_ops import Layer


class CrossEntropyLoss(Layer):
    """
        Used in classification problems mainly, will need logits from softmax
        :return crossentropy loss function
    """
    def __init__(self, softmax_logits, labels):

        super(CrossEntropyLoss, self).__init__([softmax_logits, labels])

        # it's just a real number!!!
        self.shape = (1)


    def compute(self):

        softmax_out, labels = self.prev_nodes[0].output, self.prev_nodes[1].output
        # print(softmax_out.shape, labels.shape)
        # print(type(softmax_out))
        m = self.prev_nodes[0].shape[0]
        # print(m)
        log_likelihood = -np.log(softmax_out[range(m), labels] + 0.001)  # prevent dividing by zero error, use "epsilon"
        self.output = 1 / m * np.sum(log_likelihood)
        # regularization = 0
        # loss += regularization
        return self.output









