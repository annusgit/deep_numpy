

from __future__ import print_function
from __future__ import division

# from graph_and_ops import GRAPH
from Operations import*
from utils import Data


def main():
    # generate some dummy data
    manager = Data()
    num_examples = 10**4
    max_val = 1
    train_batch_size = 32
    train_size = int(num_examples/2)
    eval_size = int(num_examples/2)
    X, y = manager.create_data_set(num_of_examples=num_examples, max_val=max_val,
                                   discriminator=lambda x: max_val*(1/(1+np.exp(-x)) + 1/(1+np.exp(x**2)))-max_val/2,
                                   one_hot=False, plot_data=True, load_saved_data=False, filename='dataset.npy')

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
    in1 = placeholder()
    in2 = placeholder()
    labels = placeholder()


    # start declaring variables
    weights1 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(100,100)))
    weights2 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(100,100)))
    weights3 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(100,100)))

    # calculate some features
    features = add(add(weights1,weights2),weights3)

    # compile your graph
    graph.graph_compile(function=features, verbose=True)


    # this is kind of sess.run()
    output = graph.run()

    # the following is an accuracy score
    print(100/10000*(np.sum(np.sum(np.add(np.add(weights1.matrix,weights2.matrix),weights3.matrix) == output))))

    pass



if __name__ == '__main__':

    main()







