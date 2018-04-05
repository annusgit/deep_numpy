

from __future__ import print_function
from __future__ import division

# from graph_and_ops import GRAPH
from Operations import*


def main():
    # start by defining your default graph
    graph = GRAPH()
    graph.getDefaultGraph()

    # start declaring variables
    weights1 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(100,100)))
    weights2 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(100,100)))
    weights3 = Matrix(initial_value=np.random.uniform(low=-0.1,high=0.1,size=(100,100)))

    features = add(add(weights1,weights2),weights3)
    graph.graph_compile(function=features, verbose=True)


    # this is kind of sess.run()
    graph.run()


    pass



if __name__ == '__main__':

    main()







