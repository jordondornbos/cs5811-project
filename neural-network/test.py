"""test.py: Test class for the neural network."""

__author__ = "Jordon Dornbos"

import example
import back_prop_learning
import multilayer_network
import re


def train(examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, weights=None, verbose=False):
    # create the network
    print 'Creating neural network'
    network = multilayer_network.MultilayerNetwork(2, num_hidden_layers, num_nodes_per_hidden_layer, 1)

    # do learning
    print 'Training neural network'
    back_prop_learning.back_prop_learning(examples, network, alpha=alpha, iteration_max=iteration_max, weights=weights,
                                          verbose=verbose)

    # print out the weights learned
    print 'Weights learned:'
    network.print_weights()
    print ''

    return network


def main():
    # array of training data
    examples = []

    # open the data file
    data = open('examples.data')

    # convert data into Example objects
    for line in data:
        line = re.sub('[\n]', '', line)  # delete newline characters

        # split the data up into x and y arrays
        values = line.split(',')
        split = len(values) - 1
        x = []
        y = []
        for val in values[0:split]:
            x.append(float(val))
        for val in values[split:len(values)]:
            y.append(float(val))

        # add the Example object
        examples.append(example.Example(x,y))

    weights = None
    network = train(examples, 0.5, 1000, 2, 3, weights)


if __name__ == '__main__':
    main()
