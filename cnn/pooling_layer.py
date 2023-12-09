from cnn.window import *

class Pooling_Layer:
    def __init__(self, section_size, stride):
        self._section_size = section_size
        self._stride = stride
    def forward(self, input):
        # Pads if needed then max pools 
        self._pad = ((input.size - self._section_size) % self._stride)
        input = input.pad(self._pad)
        self._input_size = input.size
        output_size = math.ceil((input.size - self._section_size + 1) / self._stride)
        output = []
        self.max_arr = []
        for i in range(output_size):
            output.append([])
            self.max_arr.append([])
            for j in range(output_size):
                section = input.get_section(i * self._stride, j * self._stride, self._section_size)
                output[i].append(section.max())
                max_location = section.max_location()
                max_location[0] += i * self._stride
                max_location[1] += j * self._stride
                self.max_arr[i].append(max_location)
        return Window(output)
    def backward(self, grad_output):
        # Only passes the gradient back to the max of each section as that is the only input that contributed.
        gradients = [[0 for j in range(self._input_size)] for i in range(self._input_size)]
        grad_output = grad_output.to_grid()
        for i in range(len(grad_output)):
            for j in range(len(grad_output[0])):
                location = self.max_arr[i][j]
                gradients[location[0]][location[1]] += grad_output[i][j]
        return Window(gradients).unpad(self._pad)
    def display(self):
        print("Section size: ", self._section_size)
        print("Stride: ", self._stride)
                
