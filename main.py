

"""
    A small main script to see how our library works
"""

from __future__ import print_function
from __future__ import division

# from graph_and_ops import GRAPH
from utils import Data
from Operations import*
from loss_functions import CrossEntropyLoss
from layers import Dense


def main():

    # generate some dummy data for testing
    manager = Data()
    num_examples = 10**4
    max_val = 1
    train_batch_size = 64
    train_size = int(num_examples/2)
    eval_size = int(num_examples/2)
    X, y = manager.create_data_set(num_of_examples=num_examples,
                                   max_val=max_val,
                                   discriminator=lambda x: max_val*(1/(1+np.exp(-x)) + 1/(1+np.exp(x**2)))-max_val/2,
                                   one_hot=False,
                                   plot_data=False,
                                   load_saved_data=True,
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


    # declare all the weights and biases
    weights1 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(2,32)))
    bias1 = Matrix(initial_value=np.ones(shape=32))

    weights2 = Matrix(initial_value=np.random.uniform(low=-0.1, high=0.1, size=(32, 64)))
    bias2 = Matrix(initial_value=np.ones(shape=64))

    weights3 = Matrix(initial_value=np.random.uniform(low=-0.1, high=0.1, size=(64, 128)))
    bias3 = Matrix(initial_value=np.ones(shape=128))

    weights4 = Matrix(initial_value=np.random.uniform(low=-0.1, high=0.1, size=(128, 64)))
    bias4 = Matrix(initial_value=np.ones(shape=64))

    weights5 = Matrix(initial_value=np.random.uniform(low=-0.1, high=0.1, size=(64, 32)))
    bias5 = Matrix(initial_value=np.ones(shape=32))

    weights6 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(32,2)))
    bias6 = Matrix(initial_value=np.ones(shape=2))

    # calculate some features
    features = add(dot(input_features,weights1),bias1)
    features = relu(features)
    features = add(dot(features, weights2), bias2)
    features = relu(features)
    features = add(dot(features, weights3), bias3)
    features = relu(features)
    features = add(dot(features, weights4), bias4)
    features = relu(features)
    features = add(dot(features, weights5), bias5)
    features = relu(features)

    # features = Dense(features=features, units=32)
    # features = relu(features)

    features = add(dot(features, weights6), bias6)
    logits = softmax_classifier(features)
    loss = CrossEntropyLoss(softmax_logits=logits, labels=input_labels)

    # compile and run
    graph.graph_compile(function=loss, verbose=True)
    loss = graph.run(input_matrices={input_features: train_batch_examples,
                                     input_labels: train_batch_labels})
    print(loss, logits.output.shape)


    pass



if __name__ == '__main__':

    main()







