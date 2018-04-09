


"""
    A small main script to see how our library works
"""


from __future__ import print_function
from __future__ import division

# from graph_and_ops import GRAPH
import numpy as np
from graph_and_ops import GRAPH
from Operations import placeholder, add
from layers import fully_connected, Relu
from utils import Data
from loss_functions import Softmax_with_CrossEntropyLoss


def main():

    # generate some dummy data for testing
    manager = Data()
    num_examples = 10**4
    max_val = 1
    train_batch_size = 16
    train_size = int(num_examples/2)
    eval_size = int(num_examples/2)
    X, y = manager.create_data_set(num_of_examples=num_examples, max_val=max_val,
                                   discriminator=lambda x: max_val*(1/(1+np.exp(-x)) + 1/(1+np.exp(x**2)))-max_val/2,
                                   one_hot=False, plot_data=False, load_saved_data=True,
                                   filename='dataset.npy')

    train_examples, train_labels = X[0:train_size, :], y[0:train_size]
    eval_examples, eval_labels = X[train_size:, :], y[train_size:]
    print('train examples = {}, train labels = {}, eval examples = {}, eval labels = {}'.format(train_examples.shape,
                                                                                                train_labels.shape,
                                                                                                eval_examples.shape,
                                                                                                eval_labels.shape))

    # get some small train batch
    indices = np.random.randint(low=0,high=train_size,size=train_batch_size)
    train_batch_examples, train_batch_labels = X[indices,:], y[indices]


    # start by defining your default graph
    graph = GRAPH()
    graph.getDefaultGraph()


    # declare your placeholders, to provide your inputs
    # print(int(train_batch_examples.shape[1]))
    input_features = placeholder(shape=(train_batch_size, int(train_batch_examples.shape[1])))
    input_labels = placeholder(shape=(train_batch_size))


    """
        Method #3
    """

    # this is defined using layers
    features = fully_connected(features=input_features, units=32)
    features = Relu(features)
    features = fully_connected(features=features, units=64)
    features = Relu(features)
    features = fully_connected(features=features, units=128)
    features = Relu(features)
    features_1 = fully_connected(features=features, units=256)
    features_1 = Relu(features_1)
    features_2 = fully_connected(features=features_1, units=256)
    features_2 = Relu(features_2)
    features_3 = fully_connected(features=features_2, units=256)
    features_3 = Relu(features_3)

    # check a recurrent connection
    features_3 = add(features_1, features_3)

    features = fully_connected(features=features_3, units=64)
    features = Relu(features)
    features = fully_connected(features=features, units=32)
    features = Relu(features)
    logits = fully_connected(features=features, units=2)
    loss = Softmax_with_CrossEntropyLoss(logits=logits, labels=input_labels)

    # compile and run
    graph.graph_compile(function=loss, verbose=True)
    loss = graph.run(input_matrices={input_features: train_batch_examples, input_labels: train_batch_labels})

    # run and calculate the gradients
    graph.gradients()

    print(loss, logits.output.shape)

pass


if __name__ == "__main__":
    main()
