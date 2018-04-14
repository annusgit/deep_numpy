

"""
    this file contains the actual implementations of some real operations such as multiplication, addition etc.
    for matrices
"""


from __future__ import division
from graph_and_ops import*
import numpy as np
import time
np.random.seed(int(time.time()))


class add(Operation):

    """
        our add operation; will be treated as another operation
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(add, self).__init__(input_nodes)

        # we might need their shapes at some point
        self.shape = input_nodes[0].shape
        pass


    def compute(self, **kwargs):

        # A and B are two actual matrices that we want to add
        input_matrices = [node.output for node in self.prev_nodes]
        # print(input_matrices[0])
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.add(input_matrices[0], input_matrices[1])

        return self.output


    def back(self):

        # simply save the incoming gradient (to be used by the layer behind it, so pass on to them as it is)

        # remember to add all the gradients coming from the next nodes
        # use parent class method for doing so
        super(add, self).back()

        # this will be the upstream for the previous layers
        for prev_node in self.prev_nodes:
            self.upstream_grad[prev_node] = self.upstream_grad[self]



class negative(Operation):

    """
        our negative operation
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(negative, self).__init__(input_nodes)

        # we might need their shapes at some point
        self.shape = input_nodes[0].shape
        pass


    def compute(self, **kwargs):

        # A and B are two actual matrices that we want to add
        input_matrix = [node.output for node in self.prev_nodes]
        # print(input_matrices[0])
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = -1*input_matrix

        return self.output


    def back(self):

        # multiply the incoming gradients by -1

        # remember to add all the gradients coming from the next nodes
        # use parent class method for doing so
        super(negative, self).back()

        self.upstream_grad[self.prev_nodes[0]] = -1 * self.upstream_grad[self]


class log(Operation):

    """
        our add operation; will be treated as another operation
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(log, self).__init__(input_nodes)

        # we might need their shapes at some point
        self.shape = input_nodes[0].shape
        pass


    def compute(self, **kwargs):

        # A and B are two actual matrices that we want to add
        self.input = [node.output for node in self.prev_nodes][0]
        # print(input_matrices[0])
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.log(self.input+0.001) # the 0.001 is to avoid log(0)

        return self.output


    def back(self):

        # remember to add all the gradients coming from the next nodes
        # use parent class method for doing so
        super(log, self).back()

        # so we are doing y = log(x) where x is the input, and we get dL/dy from the upstream
        # and we need dL/dx = dL/dy * dy/dx, where dy/dx = 1/x
        self.upstream_grad[self.prev_nodes[0]] = np.multiply(self.upstream_grad[self], 1/self.input)



class dot(Operation):

    """
        our dot operation; will be treated as another operation; yet to be defined
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(dot, self).__init__(input_nodes)

        # we might need their shapes at some point
        # print(input_nodes[0].shape, input_nodes[1].shape)
        self.shape = (input_nodes[0].shape[0], input_nodes[1].shape[1])
        pass


    def compute(self, **kwargs):

        # A and B are two actual matrices that we want to add
        self.input_matrices = [node.output for node in self.prev_nodes]
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.dot(self.input_matrices[0], self.input_matrices[1])

        return self.output


    def back(self):

        # get the gradients
        super(dot, self).back()
        [A, B] = self.input_matrices

        # this will be the upstream for the previous layer, dL/dA, dL/dB
        upstream_gradient = self.upstream_grad
        self.upstream_grad[self.prev_nodes[0]] = np.dot(upstream_gradient[self], B.transpose())
        self.upstream_grad[self.prev_nodes[1]] = np.dot(A.transpose(), upstream_gradient[self])
        # self.upstream_grad.append(np.dot(A, self.upstream_grad))
        # print(self.upstream_grad[0].shape, self.upstream_grad[1].shape)


class sigmoid(Operation):

    """
        our sigmoid operation; will be treated as another operation; yet to be defined
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(sigmoid, self).__init__(input_nodes)

        # we might need their shapes at some point
        # print(input_nodes[0].shape, input_nodes[1].shape)
        self.shape = input_nodes[0].shape
        pass


    def compute(self, **kwargs):

        # A and B are two actual matrices that we want to add
        x = self.prev_nodes[0].output
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = 1 / (1 + np.exp(-x))

        return self.output


    def back(self):

        super(sigmoid, self).back()

        # gradient of sigmoid
        self.upstream_grad[self.prev_nodes[0]] = self.upstream_grad[self] * self.output * (1-self.output)



class relu(Operation):

    """
        our Relu operation; will be treated as another layer; yet to be defined
    """

    def __init__(self, *input_nodes):

        # print('inside init')
        super(relu, self).__init__(input_nodes)

        # we might need their shapes at some point
        # print(input_nodes[0].shape, input_nodes[1].shape)
        self.shape = input_nodes[0].shape
        pass


    def compute(self, **kwargs):

        # we will need this input at backprop
        self.input = self.prev_nodes[0].output

        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.mask = self.input > 0 # used for backprop
        self.output = np.multiply(self.input, self.mask)

        return self.output


    def back(self):

        # get the upstream gradient using the parent method
        super(relu, self).back()

        #print(upstream_grad.shape)
        self.upstream_grad[self.prev_nodes[0]] = np.multiply(self.upstream_grad[self], self.mask)



# this one is not working at the moment
class softmax_classifier(Operation):

    """
        our softmax squashing operation to convert numbers in real probabilities
    """

    def __init__(self, *input_nodes):

        super(softmax_classifier, self).__init__(input_nodes)

        # we might need their shapes at some point
        self.shape = input_nodes[0].shape
        pass


    def compute(self, **kwargs):

        # input_matrix is the computation of the last layer before softmax
        # print(len(self.prev_nodes))
        input_matrix = self.prev_nodes[0].output

        # exp sum with numeric stability
        exps = np.exp(input_matrix-np.max(np.max(input_matrix)))
        # print(exps.shape)
        self.output = exps / np.sum(exps, axis=1)[:,None]
        # print(self.output.shape)
        return self.output


    def back(self):
        for node in self.next_nodes:
            for n in node.prev_nodes:
                if type(n).__name__ == 'placeholder':
                    true_labels_matrix = n.output

        # get the labels from the next node's prev nodes!
        # true_labels_matrix = self.next_nodes[0]

        softmax = self.output
        self.gradients = None #(upstream_grad - np.reshape(np.sum(upstream_grad * softmax, 1),[-1, 1])) * softmax

        # self.gradients = self.output * true_labels_matrix
        return self.gradients














