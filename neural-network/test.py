"""test.py: Test class for the neural network."""

__author__ = "Jordon Dornbos"

import example
import back_prop_learning
import multilayer_network
import re
import logging
from random import shuffle

LOG_FILENAME = 'neural-network.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def build_map(file, break_word=None):
    map = {}

    # put data in maps
    for line in file:
        if break_word is not None and break_word in line:
            break

        line = re.sub('[\n]', '', line)  # delete newline characters
        values = line.split(',')  # split the data up

        # map[value] = Total Flights, Delayed Flights, Rate
        logging.debug('Putting [{0} = {1}, {2}, {3}] into the map'.format(values[0], values[1], values[2], values[3]))
        map[values[0]] = [values[1], values[2], values[3]]

    return map


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


def get_data(filename, time_map, distance_map, carrier_map, airport_map):
    # open the data file
    file = open(filename)

    # skip the header
    next(file)

    # convert data file into Example objects
    data = []
    for line in file:
        line = re.sub('[\n]', '', line)  # delete newline characters
        values = line.split(',')  # split the data up

        # only consider flights that haven't been canceled
        if float(values[21]) != 1.0:
            x = []
            y = []

            try:
                # get normalized inputs
                dept_time = float(time_map[values[5][:-2]][2])
                carrier = float(carrier_map[values[8]][2])
                airport = float(airport_map[values[16]][2])

                # get delay and apply threshold
                delay = float(values[15])
                if delay > 15:
                    delay = 1.0
                else:
                    delay = 0.0

                # get distance and find normalized value from range
                distance = float(values[18])
                if distance < 300:
                    distance = float(distance_map['0'][2])
                elif 300 <= distance < 600:
                    distance = float(distance_map['1'][2])
                elif 600 <= distance < 900:
                    distance = float(distance_map['2'][2])
                else:
                    distance = float(distance_map['3'][2])

                # put the values in the arrays
                x.append(dept_time)
                x.append(carrier)
                x.append(airport)
                x.append(distance)
                y.append(delay)

            except ValueError:
                logging.debug('Could not add {0} because of an invalid value'.format(values))
                continue
            except KeyError:
                logging.debug('Could not find key for value in {0}'.format(values))
                continue

            # add the data point
            logging.debug('Adding data point [{0}, {1}]'.format(x, y))
            data.append(example.Example(x, y))

        else:
            logging.debug('Found a canceled flight')

    return data


def order(training_data):
    ordered = []

    pos = 0
    neg = 0
    data_len = len(training_data)
    pick_pos = True
    while pos < data_len and neg < data_len:
        if pick_pos:
            if training_data[pos].y[0] == 1.0:
                logging.debug('Adding positive set')
                ordered.append(training_data[pos])
                pick_pos = False
            pos += 1
        else:
            if training_data[neg].y[0] == 0.0:
                logging.debug('Adding negative set')
                ordered.append(training_data[neg])
                pick_pos = True
            neg += 1

    return ordered


def train(examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, weights=None, verbose=False):
    # create the network
    logging.info('Creating neural network...')
    network = multilayer_network.MultilayerNetwork(len(examples[0].x), num_hidden_layers, num_nodes_per_hidden_layer,
                                                   len(examples[0].y))

    # do learning
    logging.info('Training neural network...')
    hypothesis_network = back_prop_learning.back_prop_learning(examples, network, alpha=alpha,
                                                               iteration_max=iteration_max, weights=weights,
                                                               verbose=verbose)

    # print out the weights learned
    logging.info('Weights learned: {0}'.format(hypothesis_network.network.weight_string()))

    return hypothesis_network


def test(network, verification_data, verbose=False):
    logging.info('Testing accuracy...')
    num_delay_correct = 0
    num_delay_incorrect = 0
    num_on_time_correct = 0
    num_on_time_incorrect = 0
    for test in verification_data:
        # get guess and actual value
        output = network.guess(test.x)[0]
        actual = test.y[0]
        if verbose:
            logging.info('Output: {0:.3f} Actual: {1}'.format(output, actual))

        # apply threshold
        if output > 0.5:
            output = 1.0
        else:
            output = 0.0

        # add to statistics
        if output == actual:
            if actual == 1.0:
                num_delay_correct += 1
            else:
                num_on_time_correct += 1
        else:
            if actual == 1.0:
                num_delay_incorrect += 1
            else:
                num_on_time_incorrect += 1

    logging.info('Number of correct delayed flight predictions was: ' + str(num_delay_correct))
    logging.info('Number of incorrrect delay flight predictions was: ' + str(num_delay_incorrect))
    logging.info('Number of correct on time flight predictions was: ' + str(num_on_time_correct))
    logging.info('Number of incorrect on time flight predictions was: ' + str(num_on_time_incorrect))

    average_error = float(num_delay_incorrect + num_on_time_incorrect) / \
                    (num_delay_correct + num_delay_incorrect + num_on_time_correct + num_on_time_incorrect)
    average_accuracy = 1.0 - average_error
    logging.info('Average accuracy was: {0:.3f}'.format(average_accuracy))
    logging.info('Average error was: {0:.3f}'.format(average_error))

    return average_accuracy


def main():
    # get training and verification data
    logging.info('Loading data...')
    normalized_data = get_normalized_data('../data/normalized/2004_output.txt')
    training_data = get_data('../data/flight/2004_subset.csv', normalized_data[0], normalized_data[1],
                             normalized_data[2], normalized_data[3])
    normalized_data = get_normalized_data('../data/normalized/2007_output.txt')
    verification_data = get_data('../data/flight/2007_subset.csv', normalized_data[0], normalized_data[1],
                                 normalized_data[2], normalized_data[3])

    # reorganize data to avoid a bias
    shuffle(training_data)
    training_data = order(training_data)
    verification_data = order(verification_data)

    # best parameters
    best_accuracy = 0.0
    best_layers = 0
    best_nodes = 0

    # train and test the network with different parameters
    for layer in range(1, 4):
        for nodes in range(3, 11):
            logging.info('Testing with {0} layer(s) and {1} nodes per layer'.format(layer, nodes))
            network = train(training_data, 0.3, 10000, layer, nodes)
            accuracy = test(network, verification_data)

            # remember the best parameters
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layers = layer
                best_nodes = nodes

    logging.info(
        'Best accuracy ({0:.4f}) was achieved with {1} layer(s) and {2} nodes per layer'.format(accuracy, best_layers,
                                                                                                best_nodes))


if __name__ == '__main__':
    main()
