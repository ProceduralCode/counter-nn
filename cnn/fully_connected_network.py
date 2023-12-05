from fully_connected_layer import Fully_Connected_Layer
from fully_connected_util import *

class Fully_Connected_Network:
    def __init__(self, layer_sizes):
        self._layers = []
        self._activations = []
        for i in range(len(layer_sizes) - 1):
            self._layers.append(Fully_Connected_Layer(layer_sizes[i], layer_sizes[i + 1]))
    def forward(self, inputs):
        self._activations = [inputs]
        for i in range(len(self._layers)):
            self._activations.append(activation_function(self._layers[i].forward(self._activations[-1])))
        return self._activations[-1]
    def backward(self, truth):
        deltas = []
        initial_delta = []
        for i in range(len(truth)):
            initial_delta.append([(self._activations[-1][i] - truth[i]) * derived_activation(self._activations[-1][i])])
        deltas.append(initial_delta)
        for i in reversed(range(len(self._layers))):
            deltas.append(self._layers[i].get_delta(deltas[-1], self._activations[i]))
        return self._update(list(reversed(deltas)))
    def _update(self, deltas):
        for i in range(len(self._layers)):
            self._layers[i].update(deltas[i + 1], self._activations[i])
        return deltas[0]
    def display(self):
        print("Network:")
        for i in range(len(self._layers)):
            print("  Layer ", i + 1, ":")
            self._layers[i].display()
