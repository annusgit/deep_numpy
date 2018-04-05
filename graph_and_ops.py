


"""
    this file contains just a few very basic wrappers for our graph
"""

from utils import get_ordered_list


class GRAPH(object):

    def __init__(self):

        # these three will be the components associated with each graph
        self.placeholders = []
        self.Matrices = []
        self.operations = []
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
        :return: None
                 Simply makes the graph ready to work!!!

        """
        self.forward_feed_order = get_ordered_list(thisNode=function, _class=Operation)

        if verbose:
            print('log: a very crude Summary of your graph...')
            for step in self.forward_feed_order:
                print('\t {}'.format(step))


    def run(self):
        """
            this is our feed forward implementation
        """

        for step in self.forward_feed_order:
            step.compute()
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

        # these will be our own next nodes
        self.next_nodes = []

        # and this will be the output of each operation, single matrix at most!!!
        self.output = None
        pass


    def compute(self):
        "this function is called when we actually want the graph to run"
        # this method will be overridden by each child operation

        pass



class placeholder(Operation):

    """
        our input placeholder definition; will be treated as another operation
    """

    def __init__(self):

        super(placeholder, self).__init__([])
        # add it to the default graph
        default_graph.placeholders.append(self)

        pass


    def compute(self):

        pass



class Matrix(Operation):

    """
        our input placeholder definition; will be treated as another operation
    """

    def __init__(self, initial_value):

        super(Matrix, self).__init__([])
        # add it to the default graph
        default_graph.Matrices.append(self)

        # this will store our actual matrix
        self.matrix = initial_value

        pass


    def compute(self):
        # no need to return anything, just assign the value to the output
        self.output = self.matrix








