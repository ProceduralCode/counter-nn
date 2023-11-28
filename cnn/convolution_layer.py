from .window import Window

class Convolution_Layer:
    def __init__(self, filter, stride):
        self._filter = filter
        self._stride = stride
    def conv(self, input):
        input.pad(((input.size - self._filter.size) % self._stride))
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
