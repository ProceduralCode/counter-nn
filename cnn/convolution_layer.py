from window import *
import random

LEARNING_RATE = 0.1

class Convolution_Layer:
    def __init__(self, filter, stride, bias = None):
        if isinstance(filter, int):
            filter_size = filter
            filter = []
            for i in range(filter_size):
                filter.append([])
                for j in range(filter_size):
                    filter[i].append(random_weight())
        if not isinstance(filter, Window):
            filter = Window(filter)
        if bias == None:
            bias = 0
        self._filter = filter
        self._stride = stride
        self._bias = bias
    def conv(self, input):
        self._pad = (input.size - self._filter.size) % self._stride
        input = input.pad(self._pad)
        self._last_input = input
        output_size = ((input.size - self._filter.size) // self._stride) + 1
        output = []
        # Add check to make sure the above is a whole number
        for i in range(output_size):
            output.append([])
            for j in range(output_size):
                section = input.get_section(i * self._stride, j * self._stride, self._filter.size)
                section = section.multiply(self._filter)
                output[i].append(section.sum())
        return Window(output)
    def forward(self, input):
        self._last_act_input = self.conv(input).add(self._bias)
        return self._last_act_input.ReLU()
    def backward(self, grad_output):
        grad_output = self._last_act_input.dReLU().multiply(grad_output)
        grad_filter = Convolution_Layer(grad_output.pad_dilate(0, self._stride - 1), 1).conv(self._last_input)
        grad_filter = grad_filter.multiply(LEARNING_RATE)
        self._filter = self._filter.add(grad_filter)
        grad_input = Convolution_Layer(self._filter.rot180(), 1).conv(grad_output.pad_dilate(self._filter.size - 1, self._stride - 1))
        self._bias += LEARNING_RATE * grad_output.sum()
        return grad_input.unpad(self._pad)
    def display(self):
        print("Stride = ", self._stride)
        print("Bias = ", self._bias)
        print("Filter: ")
        print(self._filter)
def random_weight():
    return random.random()
