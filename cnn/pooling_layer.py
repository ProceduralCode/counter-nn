from .window import Window

class Pooling_Layer:
    def __init__(self, section_size, stride):
        self._section_size = section_size
        self._stride = stride
    def pool(self, input):
        self._input_size = [len(input), len(input[0])]
        # Pads if needed then max pools 
        input.pad(((input.size - self._section_size) % self._stride))
        output_size = ((input.size - self._section_size) // self._stride) + 1
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
    def backwards(self, err):
        # Only passes the gradient back to the max of each section as that is the only input that contributed.
        gradients = [[0 for j in range(self._input_size[1])] for i in range(self.input_size[0])]
        for i in range(len(err)):
            for j in range(len(err[0])):
                location = self.max_arr[i][j]
                gradients[location[0]][location[1]] += err[i][j]
        return gradients
                
