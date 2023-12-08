from fully_connected_layer import Fully_Connected_Layer
from fully_connected_util import *

class Fully_Connected_Network:
    def __init__(self, layer_sizes):
        if isinstance(layer_sizes, Fully_Connected_Network):
            self.copy(layer_sizes)
            return
        self._layers = []
        self._border_activations = []
        self._bn_rescale = []
        self._prev_bn_rescale_update = []
        self._bn_shift = []
        self._prev_bn_shift_update = []
        for i in range(len(layer_sizes) - 1):
            self._layers.append(Fully_Connected_Layer(layer_sizes[i], layer_sizes[i + 1]))
            self._bn_rescale.append([])
            self._prev_bn_rescale_update.append([])
            self._bn_shift.append([])
            self._prev_bn_shift_update.append([])
            for j in range(layer_sizes[i + 1]):
                self._bn_rescale[i].append(1)
                self._prev_bn_rescale_update[i].append(0)
                self._bn_shift[i].append(0)
                self._prev_bn_shift_update[i].append(0)
        return
    def forward(self, inputs):
        self._border_activations = [inputs]
        self._weight_activations = []
        self._act_fun_activations = []
        for i in range(len(self._layers)):
            self._weight_activations.append(self._layers[i].forward(self._border_activations[-1]))
            self._border_activations.append(activation_function(self._weight_activations[-1]))
            """
            self._weight_activations.append(self._layers[i].forward(self._border_activations[-1]))
            self._act_fun_activations.append(activation_function(self._weight_activations[-1]))
            self._border_activations.append(batch_normalization(self._act_fun_activations[-1], self._bn_rescale[i], self._bn_shift[i]))
            """
        return self._border_activations[-1]
    def backward(self, output_gradient):
        if not isinstance(output_gradient, list):
            output_gradient = [output_gradient]
        border_gradients = [output_gradient]
        batch_norm_gradients = []
        act_gradients = []
        # Get gradients
        for i in reversed(range(len(self._layers))):
            act_gradients.append(op(derived_activation(self._weight_activations[i]), border_gradients[-1], "*"))
            border_gradients.append(self._layers[i].get_gradient(act_gradients[-1]))
            """
            batch_norm_gradients.append(op(derived_batch_norm(self._act_fun_activations[i], border_gradients[-1], self._bn_rescale[i]), border_gradients[-1], "*"))
            act_gradients.append(op(derived_activation(self._weight_activations[i]), batch_norm_gradients[-1], "*"))
            border_gradients.append(self._layers[i].get_gradient(act_gradients[-1]))
            """
        # Reverse gradients so updating can be done forwards
        batch_norm_gradients.reverse()
        act_gradients.reverse()
        border_gradients.reverse()
        # Update perceptron layers
        for i in range(len(self._layers)):
            self._layers[i].update(act_gradients[i], self._border_activations[i])
            """
        # Update batch normalization vars
        for i in range(len(self._bn_rescale)):
            momentum = op(MOMENTUM_FACTOR, self._prev_bn_rescale_update[i], "*")
            current_term = op(LEARNING_RATE, sum(op(get_x_hat(self._act_fun_activations[i]), border_gradients[i + 1], "*")), "*")
            self._prev_bn_rescale_update[i] = op(momentum, current_term, "+")
        self._bn_rescale = op(self._bn_rescale, self._prev_bn_rescale_update, "-")
        for i in range(len(self._bn_shift)):
            momentum = op(MOMENTUM_FACTOR, self._prev_bn_shift_update[i], "*")
            current_term = op(LEARNING_RATE, sum(border_gradients[i + 1]), "*")
            self._prev_bn_shift_update[i] = op(momentum, current_term, "+")
        self._bn_shift = op(self._bn_shift, self._prev_bn_shift_update, "-")
        """
        # Returns gradient of the input
        return sum_columns(border_gradients[0])
    def display(self):
        print("Network:")
        for i in range(len(self._layers)):
            print("  Layer ", i + 1, ":")
            self._layers[i].display()
            print("     rescale = ", self._bn_rescale)
            print("     shift = ", self._bn_shift)