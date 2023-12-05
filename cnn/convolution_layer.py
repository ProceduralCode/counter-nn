from window import Window

LEARNING_RATE = 0.1

class Convolution_Layer:
    def __init__(self, filter, stride):
        self._filter = filter
        self._stride = stride
    def conv(self, input):
        self._pad = (input.size - self._filter.size) % self._stride
        input.pad(self._pad)
        self._last_input = input
        output_size = ((input.size - self._filter.size) // self._stride) + 1
        output = []
        # Add check to make sure the above is a whole number
        for i in range(output_size):
            output.append([])
            for j in range(output_size):
                section = input.get_section(i * self._stride, j * self._stride, self._filter.size)
                section.multiply(self._filter)
                output[i].append(section.sum())
        output = Window(output)
        output.ReLU
        return output
    def backward(self, grad_output):
        grad_filter = Convolution_Layer(grad_output.pad_dilate(0, self._stride - 1), 1).conv(self._last_input)
        self._filter.add(grad_filter.scalar_multiply(LEARNING_RATE))
        grad_input = Convolution_Layer(self._filter.rot180(), 1).conv(grad_output.pad_dilate(self._filter.size - 1, self._stride - 1))
        return grad_input
