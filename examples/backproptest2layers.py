


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
    train_batch_size = 32
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
    layer1 = fully_connected(features=input_features, units=32)
    layer1 = relu(layer1)
    layer2 = fully_connected(features=layer1, units=64)
    layer2 = relu(layer2)
    layer2_1 = fully_connected(features=layer2, units=64)
    layer2_1 = relu(layer2_1)
    layer2_2 = fully_connected(features=layer2_1, units=64)
    layer2_2 = relu(layer2_2)

    # a recurrent connection
    layer2_2 = add(layer2_2, layer2)

    layer3 = fully_connected(features=layer2_2, units=128)
    layer3 = relu(layer3)
    layer4 = fully_connected(features=layer3, units=128)
    layer4 = relu(layer4)
    layer5 = fully_connected(features=layer4, units=128)
    layer5 = relu(layer5)

    # a recurrent connection
    layer5 = add(layer5, layer3)

    layer6 = fully_connected(features=layer5, units=64)
    layer6 = relu(layer6)
    layer6_1 = fully_connected(features=layer6, units=64)
    layer6_1 = relu(layer6_1)
    layer6_2 = fully_connected(features=layer6_1, units=64)
    layer6_2 = relu(layer6_2)

    # a recurrent connection
    layer6_2 = add(layer6_2, layer6)

    layer7 = fully_connected(features=layer6_2, units=32)
    layer7 = relu(layer7)
    logits = fully_connected(features=layer7, units=2)
    loss = Softmax_with_CrossEntropyLoss(logits=logits, labels=input_labels)

    # compile and run
    graph.graph_compile(function=loss, verbose=True)

    # run a training loop

    # all_W = []
    # for layer in graph.forward_feed_order:
    #     if layer.is_trainable:
    #         all_W.append([layer.W, layer.bias])

    def evaluate(batch_examples, batch_labels):
        loss = graph.run(input_matrices={input_features: batch_examples, input_labels: batch_labels})
        accuracy = 100 / train_batch_size * np.sum(batch_labels == np.argmax(np.exp(logits.output) /
                                                                                    np.sum(np.exp(logits.output),
                                                                                    axis=1)[:, None], axis=1))

        return [loss, accuracy]

    def training_loop(iterations):
        for m in xrange(iterations):

            # get some small train batch
            indices = np.random.randint(low=0, high=train_size, size=train_batch_size)
            train_batch_examples, train_batch_labels = X[indices, :], y[indices]

            # get the system outputs
            [loss, accuracy] = evaluate(batch_examples=train_batch_examples, batch_labels=train_batch_labels)

            if m % 500 == 0:

                # print('-----------')
                print('log: at iteration #{}, train batch loss = {}, train batch accuracy = {}%'.format(m, loss, accuracy)) #, logits.output.shape)
                # print('batch accuracy = {}%'.format(m, accuracy)) #, logits.output.shape)
                # print('-----------')

            # calculate some evaluation accuracy
            if m != 0 and m % 2000 == 0:
                print('\n---------Evaluating Now-----------')
                eval_loss, eval_accuracy = (0, 0)
                steps = eval_size // train_batch_size
                for k in range(steps):
                    eval_indices = range(k*train_batch_size,(k+1)*train_batch_size)
                    eval_batch_loss, eval_batch_accuracy = evaluate(batch_examples=eval_examples[eval_indices,:],
                                                                    batch_labels=eval_labels[eval_indices])
                    eval_loss += eval_batch_loss
                    eval_accuracy += eval_batch_accuracy
                print('log: evaluation loss = {}, evaluation accuracy = {}%'.format(eval_loss/steps, eval_accuracy/steps))
                print('------------------------------------\n')

            # run and calculate the gradients
            graph.gradients()

            # update the weights
            graph.update(learn_rate=1e-2)

    training_loop(iterations=100000)
    # for layer, history in zip(graph.forward_feed_order, all_W):
    #     print("{}".format("True" if layer == history else "False"))
#
pass


if __name__ == "__main__":
    main()
