


"""
    this file contains just a few very basic wrappers for our graph
"""

from __future__ import print_function
from __future__ import division

from utils import get_postordered_list


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
            or as many functions as you want
        :return: None
                 Simply makes the graph ready to work!!!

        """

        # self.forward_feed_order stores the list that will be used to propagate forward through a Graph object
        # for func in functions:
        self.forward_feed_order = get_postordered_list(thisNode=function, _class=Operation)
        self.backprop_order = list(reversed(self.forward_feed_order)) # the whole list just in reverse

        if verbose:
            # print(self.operations)
            print('log: a very crude Summary of your graph...')
            for step in self.forward_feed_order:
                print('\t {} shape = {}'.format(step, step.shape))


    def run(self, input_matrices):

        """
            this is our feed forward implementation
        """

        # input_matrices will be a dictionary containing inputs to out network
        # let's assign the placeholders their values
        for placeholder in input_matrices.keys():
            placeholder.input_ = input_matrices[placeholder]

        # go through each node (step) and do them in the right order
        for step in self.forward_feed_order:
            out = step.compute()
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


    def gradients(self):

        """
            calculates all of the gradients of the loss function w.r.t network weights
        :return: a dictionary of gradients whose keys are the weights themselves
        """

        # assign a gradient of one to the loss
        upstream_gradients = 1
        for node in self.backprop_order: # basically go in reverse leaving the last (loss) element
            # if node.is_trainable:
            #     print(node)
            upstream_gradients = node.back(upstream_grad=upstream_gradients)
            # if not isinstance(upstream_gradients, int):
                # print(type(upstream_gradients))
                # print(upstream_gradients.shape)
                # print(type(node).__name__, node.shape)
                # pass
        # pass



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
        pass


    def compute(self):
        """
            Forward Prop
            this function is called when we actually want the graph to run
            this method will be overridden by each child operation
        """
        pass


    def back(self, upstream_grad):
        """
            Backward Prop
            Each operation will have its own back method to propagate in the backwards direction
        :arg all back functions will require gradients coming from the upstream
        :return: None, just calculates and assigns gradients
        """
        self.gradients = None
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
        pass


    def compute(self):
        """
            the actual layers will override this method
        :return:
        """

        pass



    def back(self, upstream_grad):


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


    def compute(self):
        # just pass on whatever value you get
        self.output = self.input_



    def back(self, upstream_grad):

        self.gradients = upstream_grad
        return self.gradients



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


    def compute(self):
        # no need to return anything, just assign the value to the output
        self.output = self.matrix



    def back(self, upstream_grad):

        self.gradients = upstream_grad
        return self.gradients








