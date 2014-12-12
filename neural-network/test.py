"""test.py: Test class for the neural network."""

__author__ = "Jordon Dornbos"

import example
import back_prop_learning
import multilayer_network
import re
import logging

LOG_FILENAME = 'neural-network.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


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


def get_normalized_data(filename):
    # open the data file
    file = open(filename)

    # skip the time header
    next(file)

    # put normalized data in maps
    logging.debug('Getting normalized time data')
    time = build_map(file, 'distance')
    logging.debug('Getting normalized distance data')
    distance = build_map(file, 'carrier')
    logging.debug('Getting normalized carrier data')
    carrier = build_map(file, 'airports')
    logging.debug('Getting normalized airport data')
    airports = build_map(file)

    return [time, distance, carrier, airports]


def build_map(file, break_word=''):
    map = {}

    # put data in maps
    for line in file:
        if break_word in line and break_word is not '':
            break

        line = re.sub('[\n]', '', line)  # delete newline characters
        values = line.split(',')  # split the data up

        # map[value] = Total Flights, Delayed Flights, Rate
        logging.debug('Putting [{0} = {1}, {2}, {3} into the map]'.format(values[0], values[1], values[2], values[3]))
        map[values[0]] = [values[1], values[2], values[3]]

    return map


def get_data(filename, time_map, distance_map, carrier_map, airport_map):
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

        # only consider flights that haven't been canceled
        if float(values[21]) is not 1.0:
            x = []
            y = []

            try:
                month = float(values[1])
                day = float(values[2])
                day_of_week = float(values[3])
                dept_time = float(values[5])
                carrier = carrier_map[values[8]][2]
                delay = float(values[15])
                airport = airport_map[values[16]][2]
                distance = float(values[18])

                x.append(month)
                x.append(day)
                x.append(day_of_week)
                x.append(dept_time)
                x.append(carrier)
                x.append(airport)
                x.append(distance)
                if delay > 15:
                    y.append(1.0)
                else:
                    y.append(0.0)

            except ValueError:
                continue

            # add the Example object
            logging.debug('Adding data point {0}, {1}]'.format(x, y))
            data.append(example.Example(x,y))

        else:
            logging.debug('Found a canceled flight')

    return data


def main():
    # get training data and verification data
    print 'Loading data...'
    normalized_data = get_normalized_data('../data/normalized/2004_output.txt')
    training_data = get_data('../data/flight/2004_subset.csv', normalized_data[0], normalized_data[1],
                             normalized_data[2], normalized_data[3])
    normalized_data = get_normalized_data('../data/normalized/2007_output.txt')
    verification_data = get_data('../data/flight/2007_subset.csv', normalized_data[0], normalized_data[1],
                                 normalized_data[2], normalized_data[3])

    # train the network with the training set
    weights = None
    logging.info('Beginning network training')
    network = train(training_data, 0.5, 1000, 3, 5, weights, verbose=True)

    # check how accurate the network is by comparing it to the verification data
    print 'Testing accuracy...'
    total_diff = 0.0
    num_data = 0
    for test in verification_data:
        output = network.guess(test.x)[0]
        if output > 0.5:
            output = 1.0
        else:
            output = 0.0

        diff = abs(test.y[0] - output)
        logging.debug('Correct output: ' + str(test.y[0]) + ', Our output: ' + str(output) + ', Error: ' + str(diff))
        total_diff += diff
        num_data += 1

    average_error = total_diff / num_data
    print 'Average accuracy was: ' + str(1.0 - average_error)
    print 'Average error was: ' + str(average_error)


if __name__ == '__main__':
    main()
