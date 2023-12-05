from fully_connected_util import *

class Fully_Connected_Layer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._weights = []
        self._prev_updates = []
        for i in range(input_size):
            weight_set = []
            prev_update_set = []
            for j in range(output_size):
                weight_set.append(random_weight())
                prev_update_set.append(0)
            self._prev_updates.append(prev_update_set)
            self._weights.append(weight_set)
        self._biases = []
        self._prev_bias_updates = []
        for i in range(output_size):
            self._biases.append(random_weight())
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
    def get_delta(self, delta_next, layer_input):
        result = dot(delta_next, self._weights)
        dActivation = derived_activation(layer_input)
        for i in range(len(result)):
            result[i] *= dActivation[i]
        return result
    def update(self, delta, activation):
        updates = arr_to_2d(dot(arrT(delta), activation))
        delta = reduce_dimension(delta)
        for i in range(len(delta)):
            self._prev_bias_updates[i] = LEARNING_RATE * delta[i] + MOMENTUM_FACTOR * self._prev_bias_updates[i]
            self._biases[i] -= self._prev_bias_updates[i] 
        for i in range(len(updates)):
            for j in range(len(updates[i])): 
                self._prev_updates[i][j] = LEARNING_RATE * updates[i][j] + MOMENTUM_FACTOR * self._prev_updates[i][j]
                self._weights[i][j] -= self._prev_updates[i][j]
        return
    def display(self):
        print("    ", self._input_size, " -> ", self._output_size)
        print("    Biases: ")
        print("      ", self._biases)
        print("    Weights: ")
        print("      ", self._weights)
        return

