


"""
    A small main script to see how our library works
"""


from __future__ import print_function
from __future__ import division
from six.moves import xrange

# from graph_and_ops import GRAPH
import numpy as np
from graph_and_ops import GRAPH
from Operations import placeholder, add, relu as relu
from layers import fully_connected, softmax_classifier
from utils import Data
from loss_functions import Softmax_with_CrossEntropyLoss


def main():

    # generate some dummy data for testing
    manager = Data()
    num_examples = 10**4
    max_val = 1
    train_batch_size = 64
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

    # start by defining your default graph
    graph = GRAPH()
    graph.getDefaultGraph()


    # declare your placeholders, to provide your inputs
    # print(int(train_batch_examples.shape[1]))
    input_features = placeholder(shape=(train_batch_size, int(train_examples.shape[1])))
    input_labels = placeholder(shape=(train_batch_size))


    """
        Method #3
    """

    # this is defined using layers
    features = fully_connected(features=input_features, units=32)
    features = relu(features)
    features = fully_connected(features=features, units=64)
    features = relu(features)
    features = fully_connected(features=features, units=128)
    features = relu(features)
    # features_1 = fully_connected(features=features, units=256)
    # features_1 = relu(features_1)
    # features_2 = fully_connected(features=features_1, units=256)
    # features_2 = relu(features_2)
    # features_3 = fully_connected(features=features_2, units=256)
    # features_3 = relu(features_3)

    # check a recurrent connection
    # features_3 = add(features_1, features_3)

    features = fully_connected(features=features, units=64)
    features = relu(features)
    features = fully_connected(features=features, units=32)
    features = relu(features)
    logits = fully_connected(features=features, units=2)
    loss = Softmax_with_CrossEntropyLoss(logits=logits, labels=input_labels)

    # compile and run
    graph.graph_compile(function=loss, verbose=True)

    # run a training loop

    # all_W = []
    # for layer in graph.forward_feed_order:
    #     if layer.is_trainable:
    #         all_W.append([layer.W, layer.bias])

    def training_loop(iterations):
        for m in xrange(iterations):

            # get some small train batch
            indices = np.random.randint(low=0, high=train_size, size=train_batch_size)
            train_batch_examples, train_batch_labels = X[indices, :], y[indices]

            loss = graph.run(input_matrices={input_features: train_batch_examples, input_labels: train_batch_labels})
            accuracy = 100 / train_batch_size * np.sum(train_batch_labels == np.argmax(np.exp(logits.output) /
                                                                                     np.sum(np.exp(logits.output),
                                                                                     axis=1)[:,None], axis=1))
            if m % 500 == 0:

                print('-----------')
                print('log: at iteration #{}, batch loss = {}'.format(m, loss)) #, logits.output.shape)
                print('log: at iteration #{}, batch accuracy = {}%'.format(m, accuracy)) #, logits.output.shape)
                # print('-----------')


            # run and calculate the gradients
            graph.gradients()

            # update the weights
            graph.update(learn_rate=8e-3)

    training_loop(iterations=100000)
    # for layer, history in zip(graph.forward_feed_order, all_W):
    #     print("{}".format("True" if layer == history else "False"))
#
pass


if __name__ == "__main__":
    main()
