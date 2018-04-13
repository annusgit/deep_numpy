

"""
    Implements a few utility functions
"""


from __future__ import print_function
from __future__ import division
from collections import OrderedDict

import os
import numpy as np
import matplotlib.pyplot as pl


# this utility class implements an ordered set
class ordered_set(OrderedDict):

    def __init__(self):
        super(ordered_set, self).__init__()

    def add(self, val):
        self[val] = None

    def get_list(self):
        return self.keys()


def get_postordered_list(thisNode):

    # a util function to get a post-order list from a graph
    """
    :return: a post order arrangement of the network to do feed-forward
    """

    # this will be our list
    postorderedlist = ordered_set()

    def postorder(node):
        # if isinstance(node, _class):
        # print(type(node))
        for prev_node in node.prev_nodes:
            # print(type(node))
            postorder(prev_node)
        postorderedlist.add(node)

    postorder(node=thisNode)
    return postorderedlist.get_list()



class Data(object):

    """
        a small class for generating some data for testing purposes only7
    """


    def __init__(self):
        pass

    def create_data_set(self, **kwargs):
        generate_data = True
        if 'load_saved_data' in kwargs.keys():
            if kwargs['load_saved_data']:
                if os.path.exists(kwargs['filename']):
                    print('log: loading saved filename {}'.format(kwargs['filename']))
                    X, y, discriminator = np.load(kwargs['filename'])
                    generate_data = False
                else:
                    print('log: file does not exist, creating new data...')
        if generate_data:
            x_vals = 2*kwargs['max_val']*np.random.rand(kwargs['num_of_examples'])-kwargs['max_val']
            x_t = np.arange(-kwargs['num_of_examples']/2,kwargs['num_of_examples']/2)/(kwargs['num_of_examples']/10)
            # print(x_t)
            X = np.asarray((x_t, x_vals))
            discriminator = np.asarray([kwargs['discriminator'](t) for t in sorted(x_t)]).transpose()
            # print(discriminator.shape)
            # np.random.shuffle(discriminator) # just do this to get a better plot
            y = np.zeros(shape=(kwargs['num_of_examples']))
            y[x_vals > discriminator] = 1
            y = y.astype(np.int32)
        if 'save_dataset' in kwargs.keys():
            np.save(kwargs['filename'], (X, y, discriminator))
            print('log: dataset saved as {}'.format(kwargs['filename']))

        if kwargs['plot_data']:
            self.__plot(X.transpose(), discriminator)

        # must do this shuffling to make a better dataset
        full_deck = np.concatenate((X.transpose(), y.reshape((y.shape[0], 1))), axis=1)
        np.random.shuffle(full_deck)
        y = full_deck[:, 2].astype(np.int32)
        X = full_deck[:, 0:2]
        # print(full_deck.shape)

        if kwargs['one_hot']:
            y = self.__one_hot(y, 1)

        return X, y

    def __one_hot(self, arr, max_val):
        new_arr = np.zeros(shape=(arr.shape[0], max_val+1))
        new_arr[range(arr.shape[0]), arr] = 1
        return new_arr

    def __plot(self, X, disc):
        # will plot only 100 values from the dataset to show some distributions
        pl.figure('Data Distribution')
        # pl.title()
        objective_function = disc
        red = X[X[:,1] > objective_function]
        green = X[X[:,1] < objective_function]
        pl.scatter(green[:,0], green[:,1], color='g', label='positive')
        pl.scatter(red[:,0], red[:,1], color='r', label='negative')
        pl.scatter(X[:,0], disc, color='b', label='discriminator')
        pl.legend()
        pl.show()
        pass









