from pooling_layer import Pooling_Layer
from convolution_layer import Convolution_Layer
from window import Window
from fully_connected_layer import Fully_Connected_Network
import random
import math
def main():
    w = Window([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(w)
    w.dilate(3)
    print(w)
    
def mlp_logic_gates():
    input_arr = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ground_truth_arr = [[0], [1], [1], [0]]
    layer_sizes = [2, 2, 1]
    mlp = Fully_Connected_Network(layer_sizes)
    #mlp.display()
    for i in range(3):
        train(mlp, input_arr, ground_truth_arr, 10000)
        print("----------------------------------------------------------")
        print("Epoch ", i + 1, ":")
        show_test(mlp, input_arr, ground_truth_arr)
    #mlp.display()
def train(mlp, dataset, truth, epochs):
    for i in range(epochs):
        for j in random.sample(range(len(dataset)), len(dataset)):
            mlp.forward(dataset[j])
            mlp.backward(truth[j])
    return dataset
def show_test(mlp, input, ground_truth):
    squared_error_sum = 0
    for i in range(len(input)):
        print("Input: ", input[i])
        prediction = mlp.forward(input[i])[0]
        actual = ground_truth[i][0]
        squared_error_sum += (prediction - actual) * (prediction - actual)
        print("Predicted: ", prediction)
        print("Actual: ", actual)
        print()
    print("MSE = ",  squared_error_sum / len(input))


main()