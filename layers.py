

"""
    this section will define some full layers instead of operations
    all of the layers will be based on operations defined by Operations
"""

from __future__ import print_function
from __future__ import division

from Operations import*


class fully_connected(Layer):
    """
        A simple dense layer
    """
    def __init__(self, features, units):

        super(fully_connected, self).__init__([features])

        # initialize its ops
        # print(features.shape)
        self.W = np.random.uniform(low=-0.1,high=0.1,size=(features.shape[1],units))
        self.bias = np.ones(shape=(features.shape[0], units))
        # print(self.W.shape, self.bias.shape)

        # this will be our connection to the network
        # this is the shape of the matrix that will come at the output!!!
        self.shape = (features.shape[0], units)

        # will be trainable
        self.is_trainable = True


    def compute(self, **kwargs):

        # just compute the dot and then add the bias
        input_matrix = self.prev_nodes[0].output
        # print(type(input_matrices[0]), type(input_matrices[1]))
        self.output = np.add(np.dot(input_matrix, self.W), self.bias)

        return self.output


    def back(self):

        # get the upstream gradient from the parent method
        super(fully_connected, self).back()

        # these will be required at weight update
        # print(self.W.shape)
        self.weight_grad = np.dot(self.prev_nodes[0].output.transpose(), self.upstream_grad)
        # self.bias_grad = np.sum(upstream_grad, axis=0)
        self.bias_grad = self.upstream_grad

        # this will be the upstream for the previous layer
        self.upstream_grad = np.dot(self.upstream_grad, self.W.transpose())


    def update(self, lr):

        """
            this method will be used to update our weights
        :return: None
        """
        self.W += -lr * self.weight_grad
        self.bias += -lr * self.bias_grad
        # print(self.W.shape, self.weight_grad.shape)
        # print(self.bias.shape, self.bias_grad.shape, np.sum(self.upstream, axis=0).shape)
        pass



# create a dropout layer
class dropout(Layer):
    """
        The dropout layer; will be non-trainable
    """
    def __init__(self, features, drop_rate=0.5):

        super(dropout, self).__init__([features])

        # this will be our connection to the network
        # this is the shape of the matrix that will come at the output!!!
        self.shape = features.shape
        # print(features.shape, self.shape)
        self.drop_rate = drop_rate

        pass


    def compute(self, **kwargs):

        # simply come up with a mask and apply it
        # and it only happens during the training time and at test time, we simply scale the input features by that
        # probability at test time
        input_matrix = self.prev_nodes[0].output
        if kwargs['mode'] == 'train':
            self.dropout_mask = np.random.randn(*self.shape) > self.drop_rate
            self.output = np.multiply(input_matrix, self.dropout_mask)
        # print(input_matrix.shape, self.dropout_mask.shape, self.output.shape)
        elif kwargs['mode'] == 'test':
            # print('=================================> testing dropout')
            self.output = self.drop_rate * input_matrix
        return self.output


    def back(self):

        # get the upstream gradient from the parent method
        super(dropout, self).back()
        # print(self.upstream_grad.shape)
        # this will be the upstream for the previous layer, we multiply with our dropout mask we used earlier
        self.upstream_grad = np.multiply(self.upstream_grad, self.dropout_mask)









