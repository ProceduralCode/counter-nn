from fully_connected_util import *

class Fully_Connected_Layer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._weights = []
        self._prev_weight_updates = []
        for i in range(input_size):
            weight_set = []
            prev_update_set = []
            for j in range(output_size):
                weight_set.append(random_weight(input_size))
                prev_update_set.append(0)
            self._prev_weight_updates.append(prev_update_set)
            self._weights.append(weight_set)
        self._biases = []
        self._prev_bias_updates = []
        for i in range(output_size):
            self._biases.append(0)
            self._prev_bias_updates.append(0)
    def forward(self, inputs):
        if(len(inputs) != self._input_size):
            raise Exception("Invalid number of inputs.")
        if isinstance(inputs[0], list):
            b_input = [[1]] + inputs
        else:
            b_input = [1] + inputs
        b_weights = [self._biases] + self._weights
        return arrT(dot(b_weights, arrT(b_input)))
    def get_gradient(self, gradient):
        return dot(gradient, self._weights)
    def update(self, gradients, layer_input):
        # update biases
        self._prev_bias_updates = op(op(LEARNING_RATE, gradients, "*"), op(MOMENTUM_FACTOR, self._prev_bias_updates, "*"), "+")
        self._biases = op(self._biases, self._prev_bias_updates, "+")
        # update weights
        weight_gradients = dot(arrT(gradients), layer_input)
        self._prev_weight_updates = op(op(LEARNING_RATE, weight_gradients, "*"), op(MOMENTUM_FACTOR, self._prev_weight_updates, "*"), "+")
        self._weights = op(self._weights, self._prev_weight_updates, "+")
        return
    def display(self):
        print("    ", self._input_size, " -> ", self._output_size)
        print("    Biases: ")
        print("      ", self._biases)
        print("    Weights: ")
        print("      ", self._weights)
        return

