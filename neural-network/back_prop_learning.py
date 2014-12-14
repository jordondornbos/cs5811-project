"""back_prop_learning.py: Backpropagation algorithm for learning in multilayer networks."""

__author__ = "Jordon Dornbos"

import random
import hypothesis_network
import multilayer_network
import logging


def back_prop_learning(examples, network, alpha=0.3, iteration_max=5000000, weights=None, verbose=False):
    """Backpropagation algorithm for learning in multilayer networks.

    Args:
        examples: A set of examples, each with input vector x and output vector y.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        alpha: The learning rate.
        iteration_max: The maximum amount of iterations to perform.
        weights: Starting weights to load into the network.
        verbose: Whether or not to print data values as the network learns.

    Returns:
        A hypothesis neural network.
    """

    delta = [0] * network.num_nodes()   # a vector of errors, indexed by network node

    # load weights if given, otherwise randomize weights
    if weights:
        network.load_weights(weights)
    else:
        randomize_weights(network, verbose=verbose)

    # keep learning until stopping criterion is satisfied
    for iteration in range(iteration_max):
        new_alpha = alpha * (1 - (float(iteration) / iteration_max))
        learn_loop(delta, examples, network, new_alpha)

        if verbose:
            logging.info('Neural network learning loop {0} of {1} with alpha: {2}'.format(iteration, iteration_max,
                                                                                          new_alpha))

    return hypothesis_network.HypothesisNetwork(network)


def randomize_weights(network, verbose=False, round=False):
    """Function to randomize perceptron weights.

    Args:
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        verbose: Whether or not to print out the weights that were assigned.
        round: Whether or not to round the printed weights.
    """
    for l in range(1, network.num_layers()):
        for n in range(network.get_layer(l).num_nodes):
            for w in range(len(network.get_node_in_layer(l, n).weights)):
                network.get_node_in_layer(l, n).weights[w] = random.random()

    if verbose:
        logging.info('Randomized weights: {0}'.format(network.weight_string(round)))


def learn_loop(delta, examples, network, alpha):
    """A loop representing the learning process.

    Args:
        delta: A list of all the delta values for the network.
        examples: A set of examples, each with input vector x and output vector y.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        alpha: The learning rate.
    """

    for example in examples:
        load_and_feed(example.x, network)

        # compute the error at the output
        for n in range(network.output_layer.num_nodes):
            delta[network.position_in_network(network.num_layers() - 1, n)] = \
                multilayer_network.sigmoid_derivative(network.output_layer.nodes[n].in_sum) * \
                (example.y[n] - network.output_layer.nodes[n].output)

        # propagate the deltas backward from output layer to input layer
        delta_propagation(delta, network)

        # update every weight in the network using deltas
        update_weights(delta, network, alpha)


def load_and_feed(input, network):
    """Function to load the input into the network and propagate the data through the network.

    Args:
        input: The values to input into the network.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
    """

    # propagate the inputs forward to compute the outputs
    for i in range(len(network.input_layer.nodes)):
        network.input_layer.nodes[i].output = input[i]

    # feed the values forward
    feed_forward(network)


def feed_forward(network):
    """Function to feed forward values in the network.

    Args:
        network: A multilayer network with L layers, weights W(j,i), activation function g.
    """

    for l in range(1, network.num_layers()):
        for n in range(network.get_layer(l).num_nodes):
            node = network.get_node_in_layer(l, n)

            summation = 0.0
            for i in range(node.num_inputs):
                summation += node.weights[i] * network.get_node_in_layer(l - 1, i).output
            summation += node.weights[len(node.weights) - 1]    # bias input

            network.get_node_in_layer(l, n).in_sum = summation
            network.get_node_in_layer(l, n).output = multilayer_network.sigmoid(summation)


def delta_propagation(delta, network):
    """Function for backpropagation the delta values.

    Args:
        delta: A list of all the delta values for the network.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
    """

    for l in range(network.num_layers() - 2, 0, -1):
        for n in range(network.get_layer(l).num_nodes):
            summation = 0.0
            next_layer_nodes = network.get_layer(l + 1).nodes
            for nln in range(len(next_layer_nodes)):
                summation += next_layer_nodes[nln].weights[n] * delta[network.position_in_network(l + 1, nln)]

            # "blame" a node as much as its weight
            delta[network.position_in_network(l, n)] = \
                multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum) * summation


def update_weights(delta, network, alpha):
    """Function to update the weights in the network.

    Args:
        delta: A list of all the delta values for the network.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        alpha: The learning rate.
    """

    for l in range(1, network.num_layers()):
        for n in range(network.get_layer(l).num_nodes):
            # adjust the weights
            node = network.get_node_in_layer(l, n)
            for i in range(node.num_inputs):
                node.weights[i] += alpha * network.get_node_in_layer(l - 1, i).output * \
                                   delta[network.position_in_network(l, n)]
            node.weights[len(node.weights) - 1] += alpha * delta[network.position_in_network(l, n)]   # bias input
