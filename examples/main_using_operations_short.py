

"""
    A small main script to see how our library works
"""

from __future__ import print_function
from __future__ import division

# from graph_and_ops import GRAPH
from utils import Data
from Operations import*
from loss_functions import CrossEntropyLoss


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
        Method #2
    """

    # define a small network
    hidden_units = [32, 64, 128, 256, 128, 64, 32, 2]

    all_weights, all_biases = [], []
    prev_units = train_batch_examples[0,:].shape[0] # this is the number of features in the inputs
    # print(prev_units)

    # declare all the weight and biases
    for units in hidden_units:
        all_weights.append(Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(prev_units,units))))
        all_biases.append(Matrix(initial_value=np.ones(shape=units)))
        prev_units = units


    # for weights in all_weights:
    #     print(weights.matrix.shape)

    # calculate the features
    features = input_features
    for weights, bias in zip(all_weights, all_biases):
        features = add(dot(features, weights), bias)

    # calculate the logits
    logits = softmax_classifier(features)

    # and the loss
    loss = CrossEntropyLoss(softmax_logits=logits, labels=input_labels)

    # compile your graph
    graph.graph_compile(function=loss, verbose=True)

    # this is kind of sess.run()
    loss = graph.run(input_matrices={input_features: train_batch_examples, input_labels: train_batch_labels})
    print(loss, logits.output.shape)



if __name__ == '__main__':

    main()