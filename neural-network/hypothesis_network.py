"""hypothesis_network.py: Hypothesis network class."""

__author__ = "Jordon Dornbos"

import back_prop_learning


class HypothesisNetwork(object):

    def __init__(self, network):
        self.network = network

    def guess(self, input):
        """Guess method for the hypothesis network.

        Args:
            input: The input to run though the network.

        Returns:
            The confidence of the input being in the function.
        """

        # load in the input and propagate it thought the network
        back_prop_learning.load_and_feed(input, self.network)
        output_layer = self.network.output_layer

        # put the output in an array (in case the output is multi-dimensional)
        output = []
        for i in range(output_layer.num_nodes):
            output.append(output_layer.nodes[i].output)

        return output
