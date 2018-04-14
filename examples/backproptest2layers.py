


"""
    A small main script to see how our library works
"""


from __future__ import print_function
from __future__ import division
from six.moves import xrange
from graph_and_ops import GRAPH
from Operations import placeholder, add, relu
from layers import fully_connected, dropout
from utils import Data
from loss_functions import Softmax_with_CrossEntropyLoss

# from graph_and_ops import GRAPH
import numpy as np
import matplotlib.pyplot as plot


def main():

    # generate some dummy data for testing
    manager = Data()
    num_examples = 10**4
    max_val = 4
    train_batch_size = 32
    train_size = int(num_examples/2)
    eval_size = int(num_examples/2)
    # discriminator = (1/(1+np.exp(-x)+1/(1+np.exp(x))))
    X, y = manager.create_data_set(num_of_examples=num_examples, max_val=max_val,
                                   discriminator=lambda x: max_val*(np.cos(np.sin(x**2)))-max_val/2,
                                   one_hot=False, plot_data=True, load_saved_data=False,
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
    # layer2 = dropout(features=layer2, drop_rate=0.5)
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
    # layer6 = dropout(features=layer6, drop_rate=0.2)
    layer6_1 = fully_connected(features=layer6, units=64)
    layer6_1 = relu(layer6_1)
    layer6_2 = fully_connected(features=layer6_1, units=64)
    layer6_2 = relu(layer6_2)

    # a recurrent connection
    layer6_2 = add(layer6_2, layer6)

    layer7 = fully_connected(features=layer6_2, units=32)
    layer7 = relu(layer7)
    # layer7 = dropout(features=layer7, drop_rate=0.5)
    logits = fully_connected(features=layer7, units=2)
    loss = Softmax_with_CrossEntropyLoss(logits=logits, labels=input_labels)

    # compile and run (always compile with the loss function, even if you don't use it!!!)
    graph.graph_compile(function=loss, verbose=True)

    # run a training loop

    # all_W = []
    # for layer in graph.forward_feed_order:
    #     if layer.is_trainable:
    #         all_W.append([layer.W, layer.bias])

    def evaluate(batch_examples, batch_labels, mode='train'):
        loss_val = graph.run(function=graph.loss, input_matrices={input_features: batch_examples, input_labels: batch_labels},
                             mode=mode)
        accuracy = 100 / train_batch_size * np.sum(batch_labels == np.argmax(np.exp(logits.output) /
                                                                                    np.sum(np.exp(logits.output),
                                                                                    axis=1)[:, None], axis=1))

        return [loss_val, accuracy]

    def training_loop(iterations):
        for m in xrange(iterations):

            # get some small train batch
            indices = np.random.randint(low=0, high=train_size, size=train_batch_size)
            train_batch_examples, train_batch_labels = X[indices, :], y[indices]

            # get the system outputs
            [loss_val, accuracy] = evaluate(batch_examples=train_batch_examples, batch_labels=train_batch_labels)
            # logits_val = graph.run(function=logits, input_matrices={input_features: train_batch_examples,
            #                                                         input_labels: train_batch_labels})
            # print(logits_val.shape)

            if m % 1000 == 0:

                # print('-----------')
                print('log: at iteration #{}, train batch loss = {}, train batch accuracy = {}%'.format(
                                                                m, loss_val, accuracy)) #, logits.output.shape)
                # print('batch accuracy = {}%'.format(m, accuracy)) #, logits.output.shape)
                # print('-----------')

            # calculate some evaluation accuracy
            if m != 0 and m % 20000 == 0:
                print('\n---------Evaluating Now-----------')
                eval_loss, eval_accuracy = (0, 0)
                steps = eval_size // train_batch_size
                for k in range(steps):
                    eval_indices = range(k*train_batch_size,(k+1)*train_batch_size)
                    eval_batch_loss, eval_batch_accuracy = evaluate(batch_examples=eval_examples[eval_indices,:],
                                                                    batch_labels=eval_labels[eval_indices], mode='test')
                    eval_loss += eval_batch_loss
                    eval_accuracy += eval_batch_accuracy
                print('log: evaluation loss = {}, evaluation accuracy = {}%'.format(eval_loss/steps, eval_accuracy/steps))
                print('------------------------------------\n')

            # do some testing
            if m != 0 and m % 50000 == 0:
                test_model()
            # run and calculate the gradients w.r.t to the loss function
            graph.gradients(function=loss)

            # check something
            # if m > 1:
            #     graph.gradients(function=logits)
            #     pass

            # update the weights
            graph.update(learn_rate=1e-2)

    def test_model():
        # now finally testing the model

        x_ = np.linspace(start=-max_val, stop=max_val, num=64)
        # we want to evaluate this thing
        test_set = np.asarray([(x, y) for x in x_ for y in x_], dtype=np.float64)

        all_predictions = []
        print('\n---------Testing Now-----------')
        print('log: your test set is {}'.format(test_set.shape))
        steps = 64 * 64 // train_batch_size
        for k in range(steps):
            test_indices = range(k * train_batch_size, (k + 1) * train_batch_size)
            test_logits = graph.run(function=logits, input_matrices={input_features: test_set[test_indices, :]},
                                    mode='test')
            test_logits -= np.max(test_logits)
            exps = np.exp(test_logits)
            softmaxed = exps / np.sum(exps, axis=1)[:, None]
            # print(softmaxed)
            predictions = np.argmax(softmaxed, axis=1)
            # print(predictions)
            # print(predictions)

            # fill up predictions
            all_predictions.append(predictions)

        print('------------------------------------\n')
        predictions = np.hstack(all_predictions)
        # print(predictions.shape)

        red = test_set[predictions == 0]
        green = test_set[predictions == 1]

        plot.scatter(green[:, 0], green[:, 1], color='g')
        plot.scatter(red[:, 0], red[:, 1], color='r')
        plot.show()

    training_loop(iterations=1000000)



pass


if __name__ == "__main__":
    main()









