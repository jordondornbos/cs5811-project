"""test.py: Test class for the neural network."""

__author__ = "Jordon Dornbos"

import example
import back_prop_learning
import multilayer_network
import re


def train(examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, weights=None, verbose=False):
    # create the network
    print 'Creating neural network...'
    network = multilayer_network.MultilayerNetwork(2, num_hidden_layers, num_nodes_per_hidden_layer, 1)

    # do learning
    print 'Training neural network...'
    hypothesis_network = back_prop_learning.back_prop_learning(examples, network, alpha=alpha,
                                                               iteration_max=iteration_max, weights=weights,
                                                               verbose=verbose)

    # print out the weights learned
    print '\nWeights learned:'
    hypothesis_network.network.print_weights()
    print ''

    return hypothesis_network


def get_data(filename):
    # array of Examples
    data = []

    # open the data file
    file = open(filename)

    # skip the header
    next(file)

    # convert data file into Example objects
    for line in file:
        line = re.sub('[\n]', '', line)  # delete newline characters

        # split the data up
        values = line.split(',')

        if values[21] != 1.0:
            x = []
            y = []

            try:
                x.append(float(values[1]))
                x.append(float(values[2]))
                x.append(float(values[3]))
                x.append(float(values[5]))
                # x.append(float(values[8]))
                x.append(float(values[9]))
                # x.append(float(values[10]))
                # x.append(float(values[16]))
                # x.append(float(values[17]))
                if float(values[15]) > 15:
                    y.append(1.0)
                else:
                    y.append(0.0)

            except ValueError:
                continue

            # add the Example object
            data.append(example.Example(x,y))

        else:
            print 'Found a canceled flight'

    return data


def main():
    # get training data and verification data
    print 'Loading data...'
    training_data = get_data('../data/flight/2001_subset.csv')
    verification_data = get_data('../data/flight/2007_subset.csv')

    # train the network with the training set
    weights = None
    network = train(training_data, 0.5, 1000, 3, 5, weights, verbose=True)

    # check how accurate the network is by comparing it to the verification data
    print 'Testing accuracy...'
    total_diff = 0.0
    num_data = 0
    for test in verification_data:
        output = network.guess(test.x)[0]
        diff = abs(test.y[0] - output)
        # print 'Correct output: ' + str(test.y[0]) + ', Our output: ' + str(output) + ', Error: ' + str(diff)
        total_diff += diff
        num_data += 1

    average_error = total_diff / num_data
    print 'Average error was: ' + str(average_error)


if __name__ == '__main__':
    main()
