

"""
    A small main script to see how our library works
"""

from __future__ import print_function
from __future__ import division

# from graph_and_ops import GRAPH
from Operations import*
from utils import Data


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

    # get some small train batch
    indices = np.random.randint(low=0,high=train_size,size=train_batch_size)
    train_batch_examples, train_batch_labels = X[indices,:], y[indices]


    # start by defining your default graph
    graph = GRAPH()
    graph.getDefaultGraph()


    # declare your placeholders, to provide your inputs
    input_features = placeholder()
    input_labels = placeholder()


    # start declaring variables
    weights1 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(2,128)))
    weights2 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(128)))
    # weights3 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(100,100)))

    # calculate some features
    features = add(dot(input_features,weights1),weights2)

    # calculate the logits
    logits = softmax_classifier(features)

    # compile your graph
    graph.graph_compile(function=logits, verbose=True)


    # this is kind of sess.run()
    output = graph.run(input_matrices={input_features: train_batch_examples})
    print(output.shape)
    # print(output.shape)

    # get the gradients and backpropagate
    # graph.gradients(); graph.back()


    # the following is an accuracy score
    # print(np.add(np.dot(train_batch_examples.transpose(),weights1.matrix), weights2.matrix) == output)

    pass



if __name__ == '__main__':

    main()







