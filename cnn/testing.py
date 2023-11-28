from pooling_layer import Pooling_Layer
from convolution_layer import Convolution_Layer
from window import Window
def main():
    test_arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    filter_arr = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    test_input = Window(test_arr)
    test_filter = Window(filter_arr)
    pool = Pooling_Layer(3, 2)
    conv = Convolution_Layer(test_filter, 2)
    print("Test Input")
    print(test_input)
    print("Pooled Result")
    print(pool.pool(test_input))
    print("Test Filter")
    print(test_filter)
    print("Convoluted Result")
    print(conv.conv(test_input))

main()