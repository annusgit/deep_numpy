

"""
    Implements a few utility functions
"""


from __future__ import print_function
from __future__ import division

from collections import OrderedDict

# this utility class implements an ordered set
class ordered_set(OrderedDict):

    def __init__(self):
        super(ordered_set, self).__init__()

    def add(self, val):
        self[val] = None

    def get_list(self):
        return self.keys()


def get_ordered_list(thisNode, _class):
    """
    :return: a post order arrangement of the network to do feed-forward
    """

    # this will be our list
    postorderedlist = ordered_set()

    def postorder(node):
        # if isinstance(node, _class):
        for prev_node in node.prev_nodes:
            postorder(prev_node)
        postorderedlist.add(node)

    postorder(node=thisNode)
    return postorderedlist.get_list()







