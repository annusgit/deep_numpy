

from __future__ import print_function
from __future__ import division
# from shutil import rmtree
from subprocess import call # for linux calls from python code
from matplotlib import pyplot as pl

import os
import time
import pickle # for saving the weights
import numpy as np
import numba #import jit, float64
np.random.seed(np.int64(time.time()))


class CPU:
    def __init__(self):
        pass

    @numba.jit(nopython=True, nogil=True, parallel=True)
    def add(self, x, y):
        result = np.zeros(shape=x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i][j] = x[i][j]+y[i][j]
        return result
        # return x + y

    # @jit
    # def add(self, x, y):
    #     return x+y

    @numba.jit(nopython=True, nogil=True, parallel=True)
    def multiply(self, x, y):
        result = np.zeros(shape=x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i][j] = x[i][j] * y[i][j]
        return result


    @numba.jit(nopython=True, nogil=True, parallel=True)
    def dot(self, x, y):
        result = np.zeros(shape=x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i][j] = x[i][j] * y[i][j]
        return result


class GRAPH(object):
    """
        This graph will be used to define all of the connections of the neural network
    """
    def __init__(self):

        pass

    def add_node(self, layer):

        pass

    def forward_pass(self):

        pass

    def backward_propagate(self):

        pass


# specs = []
# @numba.jitclass(specs)
class Network(object):

    def __init__(self, batch_size, feature_size):
        self.num_layers = 0
        self.batch_size = batch_size # we will need this for declaring our wieghts
        # self.weigths, self.biases = [], [] # empty list of weights and biases of each layer
        self.last_features_size = feature_size # used when adding newer layers
        self.summary_chain, self.net_chain = [], [] # summarize and propagate through the network, forward and backward
        self.total_parameters = 0
        pass

    def add_dense_layer(self, units):
        w = np.random.uniform(low=-0.1, high=0.1, size=(self.last_features_size, units))
        b = np.ones(shape=(self.batch_size, units))
        self.net_chain.append(['dense',w,b])
        num_params = (self.batch_size+self.last_features_size)*units
        self.summary_chain.append('layer-{}: Dense, units: {}, parameters: {}'.format(self.num_layers,units,num_params))
        self.total_parameters += num_params
        self.last_features_size = units
        self.num_layers += 1
        pass

    def dense_layer_derivatives(self, X, W):
        dhdX = W
        dhdW = X
        dhdb = 1
        return dhdX, dhdW, dhdb

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __relu(self, x):
        # just like max(0,x)
        # x[x > 100] = 100
        return np.multiply(x, x > 0)

    def __softmax(self, x):
        # this is a more stable implementation of the softmax function
        x -= np.max(x, axis=1)[:,None]
        exps = np.exp(x)
        # print(np.max(np.max(exps)))
        return exps / np.sum(exps,axis=1)[:,None]

    def add_dropout(self, drop_rate):
        self.net_chain.append(['dropout', drop_rate])
        num_params = 0
        self.summary_chain.append('layer-{}: Dropout, drop_rate: {}, parameters: {}'.format(self.num_layers, drop_rate,
                                                                                            num_params))
        self.total_parameters += num_params
        # self.last_features_size = units
        self.num_layers += 1
        pass

    def __dropout(self, x, p):
        # and we assume a uniform distribution
        dropout_mask = np.random.rand(*x.shape) < np.full(shape=x.shape, fill_value=p) # will be ones and zeros only!!!
        dropped_out = np.multiply(x, dropout_mask)
        # print('dropout man ... ', int(x.shape[0])*int(x.shape[1]), np.count_nonzero(x==0), np.count_nonzero(dropped_out==0))
        return (dropped_out, dropout_mask) # the mask will be needed for backprop

    def __dropout_back(self, dropout_cache):

        pass

    def activate(self, activation='sigmoid'):
        # if activation == 'sigmoid':
        #     pass
        self.net_chain.append(['{}'.format(activation), 'activation'])
        self.summary_chain.append('layer-{}: Activation, {}, parameters: 0'.format(self.num_layers, activation))
        # self.last_features_size = units
        self.num_layers += 1
        pass

    def summary(self):
        print('log: Summary...')
        for layer in self.summary_chain:
            print('\t', layer)
        print('\t total parameters = {}'.format(self.total_parameters))
        pass

    def net_chain_summary(self):
        for layer in self.net_chain:
            print(layer[0])
        pass

    def loss(self, logits, y):
        m = y.shape[0]
        log_likelihood = -np.log(logits[range(m), y]+0.001) # prevent dividing by zero error, use "epsilon"
        loss = 1/m*np.sum(log_likelihood)
        regularization = 0
        loss += regularization
        return loss

    def feed_forward(self, batch_data, mode='train', return_loss=True): # this will return the cache needed for backprop
        features, labels = batch_data # the features are the inputs
        cache = [] # we only save cache if we are training, no cache at test time!!!
        for idx, layer in enumerate(self.net_chain,1):
            if layer[0] == 'dense':
                if mode == 'train':
                    cache.append(('dense', features, layer[1], layer[2])) # we will need X, W and b,
                # save them before being changed
                features = np.dot(features, layer[1])
                features = np.add(features, layer[2])
                # print(features.shape)
            elif layer[0] == 'dropout':
                if mode == 'train':
                    features, dropout_cache = self.__dropout(x=features, p=layer[1])
                    cache.append(('dropout', dropout_cache)) # we will need the mask for backprop
                elif mode == 'test':
                    # we multiply the outputs by the probability of droping them, as suggested by hinton
                    features, _ = features*layer[1]
                    # no cache needed if we are testing
                    # cache.append(('dropout', dropout_cache))  # we will need the mask for backprop
            elif layer[0] == 'sigmoid':
                features = self.__sigmoid(x=features)
                if mode == 'train':
                    cache.append(('sigmoid', features)) # we will need its output
                # for its derivative depends on the output only
            elif layer[0] == 'relu':
                if mode == 'train':
                    cache.append(('relu', features)) # save them before applying the relu!!!
                features = self.__relu(x=features)
            elif layer[0] == 'softmax_classifier_with_crossentropy_loss_function':
                features = self.__softmax(x=features)
                if mode == 'train':
                    cache.append(('softmax_classifier_with_crossentropy_loss_function', features, labels))
        if return_loss:
            return features, cache, self.loss(logits=features, y=labels)
        else:
            return features, cache

    def __derivative_of_sigmoid(self, h):
        der = h*(1-h)
        # print('sigmoid derivative', h.shape, der.shape)
        return der

    def __derivative_of_relu(self, h):
        # der = np.zeros(shape=h.shape)
        # der[h > 0] = 1
        # print(np.count_nonzero(der < 0))
        return np.multiply(1, h > 0)

    def __derivative_of_softmax_classifier_with_cross_entropy_loss_function(self, logits, true_labels):
        # logits are assumed to be the outputs of the softmax classifier
        gradients = logits
        gradients[range(self.batch_size), true_labels] -= 1
        gradients /= self.batch_size
        # print(gradients.shape)
        return gradients

    def __gradients(self):

        pass

    def backward_pass(self, cache_list, learning_rate=1e-3):
        # we shall update the weights in a new list and assign it to the actual self.net_chain at the end
        # after reversing it!!!
        new_net_chain = []
        # print(self.net_chain)
        for idx, layer in enumerate(reversed(self.net_chain), 1):
            # start with calculating the softmax crossentropy loss
            if idx == 1:
                upstream_derivative = np.ones(shape=(self.batch_size, self.last_features_size))
            if layer[0] == 'softmax_classifier_with_crossentropy_loss_function':
                new_net_chain.append(layer)
                softmax_cache = cache_list.pop()
                gradients = self.__derivative_of_softmax_classifier_with_cross_entropy_loss_function(logits=softmax_cache[1],
                                                                                                     true_labels=softmax_cache[2])
                upstream_derivative = np.multiply(gradients, upstream_derivative)
                # print(idx, upstream_derivative.shape, gradients.shape, softmax_cache[0], layer[0], len(cache_list))
            elif layer[0] == 'dense':
                dense_cache = cache_list.pop()
                gradients = self.dense_layer_derivatives(X=dense_cache[1], W=dense_cache[2])
                # print(upstream_derivative.shape, layer[2].shape)
                # update the weights here
                weight_update = np.dot(gradients[1].transpose(), upstream_derivative)
                layer[1] += -learning_rate*weight_update
                layer[2] += -learning_rate*upstream_derivative
                new_net_chain.append(layer)
                upstream_derivative = np.dot(upstream_derivative, gradients[0].transpose())
                # print(idx, upstream_derivative.shape, gradients[0].shape, dense_cache[0], layer[0], len(cache_list))
                # print('shape of weigths: ', layer[1].shape, layer[2].shape)
                # layer[1] += -0.001*np.dot(gradients[1], upstream_derivative.transpose())
                # return
                # layer[0] +=
            elif layer[0] == 'dropout':
                new_net_chain.append(layer)
                dropout_cache = cache_list.pop()
                # gradients = self.__derivative_of_sigmoid(h=sigmoid_cache[1])
                upstream_derivative = np.multiply(upstream_derivative, dropout_cache[1])
            elif layer[0] == 'sigmoid':
                new_net_chain.append(layer)
                sigmoid_cache = cache_list.pop()
                gradients = self.__derivative_of_sigmoid(h=sigmoid_cache[1])
                upstream_derivative = np.multiply(upstream_derivative, gradients)
            elif layer[0] == 'relu':
                new_net_chain.append(layer)
                relu_cache = cache_list.pop()
                gradients = self.__derivative_of_relu(h=relu_cache[1])
                upstream_derivative = np.multiply(upstream_derivative, gradients)
                # print('max val = {}'.format(np.max(np.max(upstream_derivative))))
                # print(idx, upstream_derivative.shape, gradients.shape, sigmoid_cache[0], layer[0], len(cache_list))
        # del self.net_chain # just in case
        del self.net_chain
        new_net_chain.reverse()
        self.net_chain = list(new_net_chain)
        # print(type(self.net_chain))
        # print(len(cache_list))

    def accuracy(self, logits, true_labels):
        acc = np.sum(np.argmax(logits,axis=1) == true_labels)
        return acc


    def train_net(self, training_set, eval_set, lr=3e-4, lr_decay=0.95, decay_after=20, iterations=100,
                  show_train_status_after=100,evaluate_after=200, load_saved_net=False,
                  saved_model_name=None, save_dir=None, save_file_name=None, save_after=None):

        # load the saved model if asked for it
        # add_into =
        if load_saved_net and saved_model_name:
            self.net_chain = self.load_net(save_file_name=saved_model_name)


        # create the save directory for saving the models
        Save = False
        if save_after and save_dir and save_file_name:
            Save = True
            if os.path.exists(save_dir):
                # remove if it already exists there
                # rmtree(save_dir)
                # call('mkdir {}'.format(save_dir), shell=True)
                print('log: the folder {} already exists!!!'.format(save_dir))
            else:
                call('mkdir {}'.format(save_dir), shell=True)
                print('log: created your models directory {}'.format(save_dir))

        train_examples, train_labels = training_set
        for batch_number in range(iterations):
            random_indices = np.random.randint(0,train_examples.shape[0],self.batch_size)
            train_batch_examples, train_batch_labels= train_examples[random_indices,:], train_labels[random_indices]
            train_batch = (train_batch_examples, train_batch_labels)
            # forward_start = time.clock()
            if batch_number % show_train_status_after == 0:
                features, cache, batch_loss = self.feed_forward(batch_data=train_batch, mode='train', return_loss=True)
                batch_accuracy = self.accuracy(logits=features, true_labels=train_batch_labels)
                print('training: iteration # {}, batch_loss = {}, batch_accuracy = {}%'.format(batch_number,
                                                                                               batch_loss,
                                                                                               100*batch_accuracy/self.batch_size))
            else:
                features, cache = self.feed_forward(batch_data=train_batch, return_loss=False)
            # forward_time = time.clock() - forward_start
            # print(len(cache))
            # backward_start = time.clock()
            self.backward_pass(cache_list=cache, learning_rate=lr)
            # backward_time = time.clock() - backward_start
            # if batch_number % show_train_status_after == 0:
                # print('log: iteration = {} time elapsed for one batch: forward pass = {}, backward pass = {}'.format(
                #     batch_number, forward_time, backward_time))

            if batch_number % evaluate_after == 0 and batch_number > 0:
                self.evaluate(eval_set=eval_set)
            # nn.net_chain_summary()

            if batch_number % save_after == 0 and batch_number > 0 and Save :
                self.save_net(save_file_name='{}/{}-{}'.format(save_dir, save_file_name, batch_number))

            # update the learning rate after each batch
            if batch_number % decay_after == 0 and batch_number > 0:
                lr *= lr_decay
        pass

    def evaluate(self, eval_set):
        print('log: Evaluating now...')
        eval_examples, eval_labels = eval_set
        iterations = int(eval_examples.shape[0]/self.batch_size)
        total_correct_predictions, total_loss = 0, 0
        # print('iters', eval_examples.shape[0], self.batch_size)
        for batch_number in range(iterations):
            eval_batch_examples = eval_examples[batch_number*self.batch_size:(batch_number+1)*self.batch_size,:]
            eval_batch_labels = eval_labels[batch_number*self.batch_size:(batch_number+1)*self.batch_size]
            # if train_batch_labels.shape[0] < batch_size:
            #     break
            eval_batch = (eval_batch_examples, eval_batch_labels)
            # forward_start = time.clock()
            features, _, batch_loss = self.feed_forward(batch_data=eval_batch, return_loss=True)
            total_loss += batch_loss
            # forward_time = time.clock() - forward_start
            batch_correct_predictions = self.accuracy(logits=features, true_labels=eval_batch_labels)
            total_correct_predictions += batch_correct_predictions
            # print(len(cache))
            # if batch_number % show_status_after== 0:
            #     print('time elapsed for one batch: forward pass = {}, backward pass = {}'.format(forward_time))
        print('Evaluation loss = {}, Evaluation Accuracy = {}%'.format(total_loss/(iterations+1),
                                                                       100*total_correct_predictions/(iterations*self.batch_size)))
        pass

    def save_net(self, save_file_name):
        # we shall save our network in a pickle, will contain the net_chain, hence the entire set of weights as well!!
        with open('{}.pickle'.format(save_file_name), 'wb') as this_file:
            if os.path.exists(save_file_name):
                print('log: A model with the same name already exists! Collision!!!, not saving the new one...')
                return
            pickle.dump(obj=self.net_chain, file=this_file, protocol=pickle.HIGHEST_PROTOCOL)
            print('log: saved model {}========================================\n'.format('{}.pickle'.format(save_file_name)))
        pass

    # def load_saved_model(self, model):
        # this will call the load_net function
        # return self.load_net(model)

    def load_net(self, save_file_name):
        # this should return the net_chain
        load_file_name = '{}.pickle'.format(save_file_name) if not save_file_name.endswith('.pickle') else save_file_name
        if os.path.exists(load_file_name):
            with open(load_file_name, 'rb') as this_file:
                print('log: loading {}...'.format(load_file_name))
                self.net_chain = pickle.load(file=this_file)
        else:
            print('log: file {} does not exist!!'.format(load_file_name))
        pass


class Data(object):

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
        pl.scatter(red[:,0], red[:,1], color='r')
        pl.scatter(green[:,0], green[:,1], color='g')
        pl.scatter(X[:,0], disc, color='b', label='discriminator')
        pl.legend()
        pl.show()
        pass


def main():
    manager = Data()
    num_examples = 10**4
    max_val = 1
    train_batch_size = 32
    train_size = int(num_examples/2)
    eval_size = int(num_examples/2)
    X, y = manager.create_data_set(num_of_examples=num_examples, max_val=max_val,
                                   discriminator=lambda x: max_val*(1/(1+np.exp(-x)) + 1/(1+np.exp(x**2)))-max_val/2,
                                   one_hot=False, plot_data=True, load_saved_data=False, filename='dataset.npy')

    # examples_batch = X[:batch_size,:]
    # labels_batch = y[:batch_size]
    # test_batch = (examples_batch, labels_batch)
    # # print(test_batch[0].shape, test_batch[1].shape)

    # print(X.shape, y.shape)
    train_examples, train_labels = X[0:train_size, :], y[0:train_size]
    eval_examples, eval_labels = X[train_size:, :], y[train_size:]
    print('train examples = {}, train labels = {}, eval examples = {}, eval labels = {}'.format(train_examples.shape,
                                                                                                train_labels.shape,
                                                                                                eval_examples.shape,
                                                                                                eval_labels.shape))

    nn = Network(batch_size=train_batch_size, feature_size=X.shape[1])
    nn.add_dense_layer(units=128)
    nn.activate(activation='relu')
    nn.add_dropout(drop_rate=0.5)

    nn.add_dense_layer(units=256)
    nn.activate(activation='relu')
    # nn.add_dropout(drop_rate=0.5)

    nn.add_dense_layer(units=512)
    nn.activate(activation='relu')
    # nn.add_dropout(drop_rate=0.5)

    # nn.add_dense_layer(units=512)
    # nn.activate(activation='sigmoid')
    # nn.add_dropout(drop_rate=0.5)

    # nn.add_dense_layer(units=512)
    # nn.activate(activation='sigmoid')
    # nn.add_dropout(drop_rate=0.2)

    # nn.add_dense_layer(units=256)
    # nn.activate(activation='relu')
    # nn.add_dropout(drop_rate=0.5)

    # nn.add_dense_layer(units=128)
    # nn.activate(activation='relu')
    # nn.add_dropout(drop_rate=0.5)

    nn.add_dense_layer(units=64)
    nn.activate(activation='relu')
    # nn.add_dropout(drop_rate=0.5)

    nn.add_dense_layer(units=32)
    nn.activate(activation='sigmoid')
    nn.add_dropout(drop_rate=0.5)

    nn.add_dense_layer(units=2)
    nn.activate(activation='softmax_classifier_with_crossentropy_loss_function')
    nn.summary()
    # # nn.net_chain_summary()

    # nn.load_net(save_file_name='numpy_models/numpy_model-4999.pickle')
    # nn.summary()

    nn.train_net(training_set=(train_examples, train_labels), eval_set=(eval_examples, eval_labels),
                 lr=5e-3, lr_decay=0.95, decay_after=5000, iterations=10**6, show_train_status_after=100,
                 load_saved_net=False, saved_model_name='numpy_models/numpy_model-4999.pickle', evaluate_after=500,
                 save_dir='numpy_models', save_file_name='numpy_model', save_after=5000)

    # for i in range(10**4):
    #     forward_start = time.clock()
    #     if i % 100 == 0:
    #         feats, cache = nn.feed_forward(batch_data=test_batch, show_loss=True)
    #     else:
    #         feats, cache = nn.feed_forward(batch_data=test_batch, show_loss=False)
    #     forward_time = time.clock()-forward_start
    #     # print(len(cache))
    #     backward_start = time.clock()
    #     nn.backward_pass(cache_list=cache)
    #     backward_time = time.clock() - backward_start
    #     if i % 100 == 0:
    #         print('time elapsed: forward pass = {}, backward pass = {}'.format(forward_time, backward_time))
    #     # nn.net_chain_summary()
    #     # nn.summary()


    # print(np.sum(feats, axis=1))


if __name__ == "__main__":
    main()







