


"""
    this file contains just a few very basic wrappers for our graph
"""

from __future__ import print_function
from __future__ import division

from utils import get_postordered_list
import numpy as np


class GRAPH(object):

    def __init__(self):

        # these three will be the components associated with each graph
        self.placeholders = []
        self.Matrices = []
        self.operations = []
        self.layers = []
        self.losses = []

        pass


    def getDefaultGraph(self):

        # so basically all of the operations will be associated with this one graph
        # and we shall use it to forward and backward propagate through the network

        global default_graph
        default_graph = self
        print('log: Using default graph...')


    def graph_compile(self, function, verbose=False):
        """
            get a post-order of the graph for feed forward, needs a function to target for feed-forward
            the target should always be the loss function
        :return: None
                 Simply makes the graph ready to work!!!

        """

        # self.forward_feed_order stores the list that will be used to propagate forward through a Graph object
        # for func in functions:
        loss_forward_feed_order = get_postordered_list(thisNode=function)
        loss_backprop_order = list(reversed(loss_forward_feed_order)) # the whole list just in reverse
        self.loss = function

        # also maintain a dict of operations that we can perform
        self.forward_propagation_dict, self.backward_propagation_dict = {}, {}
        self.forward_propagation_dict[self.loss] = loss_forward_feed_order
        self.backward_propagation_dict[self.loss] = loss_backprop_order

        if verbose:
            # print(self.operations)
            print('log: a very crude Summary of your graph...')
            for step in self.forward_propagation_dict[self.loss]:
                print('\t {} shape = {}'.format(type(step).__name__, step.shape))

            # print('log: And this will be the order of backprop...')
            # for step in self.backward_propagation_dict[self.loss]:
            #     print('\t {} shape = {}'.format(step, step.shape))


    def run(self, function, input_matrices, mode='train'):

        """
            this is our feed forward implementation
        """

        # input_matrices will be a dictionary containing inputs to out network
        # let's assign the placeholders their values
        for placeholder in input_matrices.keys():
            placeholder.input_ = input_matrices[placeholder]

        # go through each node (step) and do them in the right order
        # but find the function first
        if function in self.forward_propagation_dict.keys():
            forward_order = self.forward_propagation_dict[function]
        else:
            # get it's post order
            self.forward_propagation_dict[function] = get_postordered_list(thisNode=function)
            forward_order = self.forward_propagation_dict[function]

        for step in forward_order:
            # mode will be helpful for operations like dropout
            out = step.compute(mode=mode)
            # print(out.shape)

        # return the final output
        return out


    def back_propagate(self):

        """
            apply backward prop on our network, will assume that the gradients have been calculated
            will simply update all of the network weights
            call this method when the gradients have been calculated
        :return: None
        """

        pass


    def gradients(self, function):

        """
            calculates all of the gradients of the loss function w.r.t network weights
        :return: a dictionary of gradients whose keys are the weights themselves
        """

        # assign a gradient of one to the derivative of loss w.r.t the loss function
        # upstream_gradients = 1
        # self.backprop_order[-1].upstream_grad = 1

        # get that function
        if function in self.backward_propagation_dict.keys():
            back_order = self.backward_propagation_dict[function]
        else:
            forward_order = get_postordered_list(thisNode=function)
            back_order = list(reversed(forward_order))

        for node in back_order: # basically go in reverse leaving the last (loss) element
            # if node.is_trainable:
            #     print(node)
            # if not isinstance(node, Matrix) and not isinstance(node, placeholder):
                # print(node)
            node.back()
            # print(type(node).__name__, upstream_gradients)
            # if not isinstance(upstream_gradients, int):
                # print(type(upstream_gradients))
                # print(upstream_gradients.shape)
                # print(type(node).__name__, node.shape)
                # pass
        # pass

    def update(self, learn_rate=3e-4):

        for node in self.backward_propagation_dict[self.loss]:
            if node.is_trainable:
                node.update(lr=learn_rate)
        """
            this method will update the weights using the gradients
        :return:
        """

        pass



class Operation(object):

    """
        this class will contain a very basic parent class for all types of operations
    """

    def __init__(self, inputs):
        " this function will be called when we are defining the graph"
        self.prev_nodes = inputs

        # tell the inputs that we are going to "consume" them!!!
        for prev_node in self.prev_nodes:
            prev_node.next_nodes.append(self)


        # add to the graph
        default_graph.operations.append(self)

        # these will be our own next nodes
        self.next_nodes = []

        # and this will be the output of each operation, single matrix at most!!!
        self.output = None

        # this will tell us which ops to train and which not to train
        self.is_trainable = False

        # store the upstream gradient, in the form of a dictionary, will be very important
        self.upstream_grad = {}

        pass


    def compute(self, **kwargs):
        """
            Forward Prop
            this function is called when we actually want the graph to run
            this method will be overridden by each child operation
        """
        pass


    def back(self):
        """
            Backward Prop
            Each operation will have its own back method to propagate in the backwards direction
        :arg all back functions will require gradients coming from the upstream
        :return: None, just calculates and assigns gradients
        """

        # remember to add all the gradients coming from the next nodes
        if len(self.next_nodes) != 0:
            self.upstream_grad[self] = np.zeros_like(self.output)
            for node in self.next_nodes:
                self.upstream_grad[self] = np.add(self.upstream_grad[self], node.upstream_grad[self])
        else: # then it must be the last node, most probably the lost function itself
            self.final_gradient = 1

        # print(self.gradients.shape, end='')
        pass



# this will look just like the Operations class
class Layer(object):

    """
        this class will contain a very basic parent class for all the layers
    """

    def __init__(self, inputs):
        " this function will be called when we are defining the graph"
        self.prev_nodes = inputs

        # tell the inputs that we are going to "consume" them!!!
        for prev_node in self.prev_nodes:
            prev_node.next_nodes.append(self)

        # add to the graph
        default_graph.layers.append(self)


        # these will be our own next nodes
        self.next_nodes = []

        # and this will be the output of each operation, single matrix at most!!!
        self.output = None

        # again, set trainable param
        self.is_trainable = False

        # store the upstream gradient
        self.upstream_grad = {}

        pass


    def compute(self, **kwargs):
        """
            the actual layers will override this method
        :return:
        """

        pass



    def back(self):

        # remember to add all the gradients coming from the next nodes
        if len(self.next_nodes) != 0:
            self.upstream_grad[self] = np.zeros_like(self.output)
            for node in self.next_nodes:
                self.upstream_grad[self] = np.add(self.upstream_grad[self], node.upstream_grad[self])
        else: # then it must be the last node, most probably the lost function itself
            self.final_gradient = 1

        pass




class placeholder(Operation):

    """
        our input placeholder definition; will be treated as another operation
    """

    def __init__(self, shape):

        super(placeholder, self).__init__([])
        # add it to the default graph
        default_graph.placeholders.append(self)

        # they need some input
        self.input_ = None

        # shape of the input
        self.shape = shape
        pass


    def compute(self, **kwargs):
        # just pass on whatever value you get
        self.output = self.input_


    # we don't need this
    def back(self):

        pass


class Matrix(Operation):

    """
        our Matrix definition; will be treated as another operation
    """

    def __init__(self, initial_value):

        super(Matrix, self).__init__([])
        # add it to the default graph
        default_graph.Matrices.append(self)

        # this will store our actual matrix
        self.matrix = initial_value

        self.shape = self.matrix.shape

        # and it will be trainable ofcourse
        self.is_trainable = True
        pass


    def compute(self, **kwargs):
        # no need to return anything, just assign the value to the output
        self.output = self.matrix


    def update(self, lr):

        super(Matrix, self).back()

        # look for the biases vs. weights issue
        try:
            self.matrix += -lr * self.upstream_grad[self]
        except ValueError:
            try:
                self.matrix += -lr * np.sum(self.upstream_grad[self], axis=0)
            except ValueError:
                print('it\'s not working!!!')
                pass
        pass



    # def back(self, upstream_grad):
    #
    #     self.gradients = upstream_grad
    #     return self.gradients








