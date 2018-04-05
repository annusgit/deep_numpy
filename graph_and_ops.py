


"""
    this file contains just a few very basic wrappers for our graph
"""


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




class Operation(object):

    """
        this class will contain a very basic parent class for all types of operations
    """

    def __init__(self, input_edges):
        " this function will be called when we are defining the graph"
        self.input_edges = input_edges
        self.output_edges = []
        pass


    def compute(self):
        "this function is called when we actually want the graph to run"
        # this method will be overridden by each child operation

        pass



class placeholder(Operation):

    """
        our input placeholder definition; will be treated as another operation
    """

    def __init__(self, *inputs):

        super(placeholder, self).__init__(inputs)
        # add it to the default graph
        default_graph.placeholders.append(self)

        pass


    def compute(self):
        pass


class Matrix(Operation):

    """
        our input placeholder definition; will be treated as another operation
    """

    def __init__(self, inputs):

        super(placeholder, self).__init__()
        # add it to the default graph
        default_graph.Matrices.append(self)
        pass


    def compute(self):
        pass













