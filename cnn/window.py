import math

BN_SMOOTH = 0.1
LEAK = 0.1

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
        if(self.size != other.size):
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
        return self
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
        if isinstance(other, Window):
            return self.copy()._multiply(other)
        else:
            return self.copy()._scalar_multiply(other)
    def _multiply(self, other):
        # Multiplies each value by the corresponding one in another instance.
        if(self.size != other.size):
            raise Exception("Filter matrix is the wrong size")
        for i in range(self.size):
            for j in range(self.size):
                self._grid[i][j] *= other._grid[i][j]
        return self
    def _scalar_multiply(self, x):
        for i in range(self.size):
            for j in range(self.size):
                self._grid[i][j] *= x
        return self
    def add(self, other):
        if isinstance(other, Window):
            return self.copy()._add(other)
        else:
            return self.copy()._scalar_add(other)
    def _add(self, other):
        # Adds each value by the corresponding one in another instance.
        if(self.size != other.size):
            raise Exception("Filter matrix is the wrong size")
        for i in range(self.size):
            for j in range(self.size):
                self._grid[i][j] += other._grid[i][j]
        return self
    def _scalar_add(self, x):
        for i in range(self.size):
            for j in range(self.size):
                self._grid[i][j] += x
        return self
    def sum(self):
        # Sums the grid's contents
        sum = 0
        for i in range(self.size):
            for j in range(self.size):
                sum += self._grid[i][j]
        return sum
    def avg(self):
        return (self.sum / self.size * self.size)
    def stdev(self):
        dif = self.add(-1 * self.avg)
        return math.sqrt(dif.multiply(dif).avg())
    def get_bn(self):
        return self.add(-1 * self.avg()).multiply(1 / math.sqrt(self.stdev() ** 2 + BN_SMOOTH))
    def bn_rescale(self, rescale, shift):
        return self.get_bn().multiply(rescale).add(shift)
    def max(self):
        # Returns the max value in the grid
        m = self._grid[0][0]
        for i in range(self.size):
            for j in range(self.size):
                m = max(m, self._grid[i][j])
        return m
    def max_location(self):
        m = self._grid[0][0]
        location = [0, 0]
        for i in range(self.size):
            for j in range(self.size):
                if m != max(m, self._grid[i][j]):
                    m = max(m, self._grid[i][j])
                    location = [i, j]
        return location
    def flatten(self):
        # Returns a flattened (1d) version 
        flat = []
        for i in range(self.size):
            for j in range(self.size):
                flat.append(self._grid[i][j])
        return flat       
    def ReLU(self):
        return self.copy()._ReLU()
    def _ReLU(self):
        # Applies the ReLU activation function to the grid.
        for i in range(self.size):
            for j in range(self.size):
                if self._grid[i][j] < 0:
                    self._grid[i][j] *= LEAK
        return self
    def dReLU(self):
        return self.copy()._dReLU()
    def _dReLU(self):
        # Applies the ReLU activation function to the grid.
        for i in range(self.size):
            for j in range(self.size):
                if self._grid[i][j] < 0:
                    self._grid[i][j] = LEAK
                else:
                    self._grid[i][j] = 1
        return self
    def sigmoid(self):
        return self.copy()._sigmoid()
    def _sigmoid(self):
        # Applies the sigmoid activation function to the grid.
        for i in range(self.size):
            for j in range(self.size):
                self._grid[i][j] = 1 / (1 + math.e ** -self._grid[i][j])
        return self
    def rot180(self):
        flipped = self.copy()
        for i in range(flipped.size):
            flipped._grid[i].reverse()
        flipped._grid.reverse()
        return flipped
    def pad(self, pad_amount):
        return self.copy()._pad(pad_amount)
    def _pad(self, pad_amount):
        # pads with pad_amount zeroes. Should be changed to spread them evenly.
        for i in range(math.floor(pad_amount / 2)):
            self._grid = self._pad_se(self._grid)
            self._grid = self._pad_nw(self._grid)
        if pad_amount % 2 == 1:
            self._grid = self._pad_se(self._grid)
        self.size += pad_amount
        return self
    def _pad_nw(self, grid):
        for i in range(len(grid)):
            grid[i].insert(0, 0)
        arr = []
        for i in range(len(grid) + 1):
            arr.append(0)
        grid.insert(0, arr)
        return grid
    def _pad_se(self, grid):
        for i in range(len(grid)):
            grid[i].append(0)
        arr = []
        for i in range(len(grid) + 1):
            arr.append(0)
        grid.append(arr)
        return grid
    def dilate(self, dilate):
        return self.copy()._dilate(dilate)
    def _dilate(self, dilate):
        for i in range(len(self._grid)):
            for j in reversed(range(len(self._grid[i]) - 1)):
                for k in range(dilate):
                    self._grid[i].insert(j + 1, 0)
        for i in reversed(range(len(self._grid) - 1)):
            for j in range(dilate):
                temp = []
                for k in range(len(self._grid[0])):
                    temp.append(0)
                self._grid.insert(i + 1, temp)
        self.size = self.size + ((self.size - 1) * dilate)
        return self
    def pad_dilate(self, pad, dilate):
        return self.dilate(dilate).pad(pad * 2)
    def unpad(self, pad_amount):
        if pad_amount > 0:
            raise Exception("unpad not implemented yet")
        return self.copy()
    def to_grid(self):
        return self.copy()._grid


def unflatten(flat):
    root = math.floor(math.sqrt(len(flat)))
    if len(flat) % root != 0:
        raise Exception("Array to unflatten isn't sqrtable")
    grid = []
    for i in range(root):
        temp = []
        for j in range(root):
            temp.append(flat[i * root + j])
        grid.append(temp)
    return Window(grid)