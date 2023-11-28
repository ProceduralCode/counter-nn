from .window import Window

class Pooling_Layer:
    def __init__(self, section_size, stride):
        self._section_size = section_size
        self._stride = stride
    def pool(self, input):
        # Pads if needed then max pools
        input.pad(((input.size - self._section_size) % self._stride))
        output_size = ((input.size - self._section_size) // self._stride) + 1
        output = []
        for i in range(output_size):
            output.append([])
            for j in range(output_size):
                section = input.get_section(i * self._stride, j * self._stride, self._section_size)
                output[i].append(section.max())
        return Window(output)


