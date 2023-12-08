from window import *

class Convolutional_Network:
    def __init__(self, conv_pool_layers, mlp):
        self._conv_pool_layers = conv_pool_layers
        self._mlp = mlp
    def forward(self, input):
        if not isinstance(input, Window):
            if isinstance(input, list) and isinstance(input[0], list) and isinstance(input[0][0], list):
                if len(input) != len(self._conv_pool_layers):
                    raise Exception("Invalid input depth")
                input = [Window(element) for element in input]
            elif not isinstance(input[0],Window):
                input = Window(input)
        if isinstance(input, list):
            input = input.copy()
        for layer in self._conv_pool_layers:
            if isinstance(layer, list):
                for i in range(len(layer)):
                    input[i] = layer[i].forward(input[i])
            else:
                input = layer.forward(input)
        if isinstance(input, list):
            test = input
            flat = [element.flatten() for element in input]
            input = []
            for arr in flat:
                for element in arr:
                    input.append(element)
        else:
            input = input.flatten()
        return self._mlp.forward(input)
    def backward(self, gradient):
        gradient = self._mlp.backward(gradient)
        if isinstance(self._conv_pool_layers[0], list):
            new_grad = []
            depth = len(self._conv_pool_layers[0])
            section = int(len(gradient) / depth)
            for i in range(depth):
                new_grad.append(unflatten(gradient[i * section : (i + 1) * section]))
            gradient = new_grad
            for layer in reversed(self._conv_pool_layers):
                for i in range(depth):
                    gradient[i] = layer[i].backward(gradient[i])
        else:
            gradient = unflatten(gradient)
            for layer in reversed(self._conv_pool_layers):
                gradient = layer.backward(gradient)
        return gradient
    def display(self):
        c_p = ["Convolutional layer: ", "Pooling layer: "]
        print("Convolutional Network:")
        for i in range(len(self._conv_pool_layers)):
            print(c_p[i % 2])
            self._conv_pool_layers[i].display()
        print()
        print("Fully connected layer: ")
        self._mlp.display()