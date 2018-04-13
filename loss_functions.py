

"""

    Here we shall implement some commonly used loss functions,
    contemporary losses may be defined using basic operations from Operations
"""


from __future__ import print_function
from __future__ import division

import numpy as np
from graph_and_ops import Layer as Loss


class CrossEntropyLoss(Loss):
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


    def back(self, upstream_grad):


        pass



class Softmax_with_CrossEntropyLoss(Loss):
    """
        Used in classification problems mainly, will need logits from softmax
        :return crossentropy loss function
    """
    def __init__(self, logits, labels):

        super(Softmax_with_CrossEntropyLoss, self).__init__([logits, labels])

        # it's just a real number!!!
        self.shape = (1)


    def compute(self):

        logits, labels = self.prev_nodes[0].output, self.prev_nodes[1].output

        # compute the softmax first
        exps = np.exp(logits - np.max(np.max(logits)))
        # print(exps.shape)
        self.softmax_logits = exps / np.sum(exps, axis=1)[:, None]


        # print(softmax_out.shape, labels.shape)
        # print(type(softmax_out))
        m = self.softmax_logits.shape[0]
        # print(m)
        log_likelihood = -np.log(self.softmax_logits[range(m), labels] + 0.001)  # prevent dividing by zero error, use "epsilon"
        self.output = 1 / m * np.sum(log_likelihood)
        # regularization = 0
        # loss += regularization
        return self.output


    def back(self):

        # gradients = logits
        # gradients[range(self.batch_size), true_labels] -= 1
        # gradients /= self.batch_size

        # get the upstream gradient
        super(Softmax_with_CrossEntropyLoss, self).back()

        # logits are assumed to be the outputs of the softmax classifier
        true_labels = self.prev_nodes[1].output
        batch_size = self.prev_nodes[0].shape[0]
        gradients = self.softmax_logits

        gradients[range(batch_size), true_labels] -= 1
        gradients /= batch_size
        # print(gradients.shape)
        self.upstream_grad =  np.multiply(gradients, self.upstream_grad)






