class Window:
    # Class to hold and manipulate 2d arrays that will be used by the convolutional and pooling layers
    def __init__(self, grid):
        self._grid = grid
        self.make_square()
        self.size = len(self._grid)
        return
    def __str__(self):
        contents = ""
        for i in range(self.size):
            for j in range(self.size):
                contents += str(self._grid[i][j]) + "  "
            contents += "\n"
        return contents
    def __eq__(self, other):
        if(self.__size != other._size):
            return False
        for i in range(self.size):
            for j in range(self.size):
                if(self._grid[i][j] != other._grid[i][j]):
                    return False
        return True
    def copy(self):
        # returns a copy.
        new_grid = []
        for i in range(self.size):
            new_grid.append([])
            for j in range(self.size):
                new_grid[i].append(self._grid[i][j])
        return Window(new_grid)
    def make_square(self):
        # Pads with 0s to make the grid square
        if(len(self._grid) > len(self._grid[0])):
            for i in range(len(self._grid)):
                for j in range(len(self._grid) - len(self._grid[0])):
                    self._grid[i].append(0)
        elif(len(self._grid) < len(self._grid[0])):
            for i in range(len(self._grid[0]) - len(self._grid)):
                new_row = []
                for j in range(len(self._grid[0])):
                    new_row.append(0)
                self._grid.append(new_row)
        return
    def pad(self, pad_amount):
        # pads with pad_amount zeroes. Should be changed to spread them evenly.
        for i in range(self.size):
            for j in range(pad_amount):
                self._grid[i].append(0)
        for i in range(pad_amount):
            new_row = []
            for j in range(self.size + pad_amount):
                new_row.append(0)
            self._grid.append(new_row)
        self.size += pad_amount
        return
    def get_section(self, x, y, size):
        # Gets a square section of the grid with side length size and a corner at (x, y)
        if(x < 0 or y < 0 or x + size > self.size or y + size > self.size):
            raise Exception("Index out of bounds")
        output = []
        for i in range(size):
            output.append([])
            for j in range(size):
                output[i].append(self._grid[x + i][y + j])
        return Window(output)
    def multiply(self, other):
        # Multiplies each value by the corresponding one in another instance.
        if(self.size != other.size):
            raise Exception("Filter matrix is the wrong size")
        for i in range(self.size):
            for j in range(self.size):
                self._grid[i][j] *= other._grid[i][j]
        return
    def sum(self):
        # Sums the grid's contents
        sum = 0
        for i in range(self.size):
            for j in range(self.size):
                sum += self._grid[i][j]
        return sum
    def max(self):
        # Returns the max value in the grid
        m = 0
        for i in range(self.size):
            for j in range(self.size):
                m = max(m, self._grid[i][j])
        return m
    def flatten(self):
        # Returns a flattened (1d) version 
        flat = []
        for i in range(self.size):
            for j in range(self.size):
                flat.append(self._grid[i][j])
        return flat
    def ReLU(self):
        # Applies the ReLU activation function to the grid.
        for i in range(self.size):
            for j in range(self.size):
                self._grid[i][j] = max(0, self._grid[i][j])