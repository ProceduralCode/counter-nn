from cnn.pooling_layer import Pooling_Layer
from cnn.convolution_layer import Convolution_Layer
from cnn.window import *
from cnn.fully_connected_network import Fully_Connected_Network
from cnn.fully_connected_util import *
from cnn.convolutional_network import Convolutional_Network
import random
import math
import numpy as np
from main import Dataset

LOGIC_GATES = [[0, 0], [0, 1], [1, 0], [1, 1]]
AxorB= [[0], [1], [1], [0]]
A = [[0], [0], [1], [1]]
FOUR_INPUTS = [element.flatten() for element in generate_bin_squares(2)]
SUM_FOUR_INPUTS = [sum(element) for element in FOUR_INPUTS]
BIN_GRID = generate_bin_squares(3)
FEATURE_FILTER = Window([[1, 0], [0, 1]])
BIN_GRID_CL_OUT = filter_features(BIN_GRID, FEATURE_FILTER)
BIN_GRID_CNN_OUT = count_features(BIN_GRID, FEATURE_FILTER)


MSE_THRESHOLD = 0.05

def main():
    bees_train(1)
    """
    print("Testing a multi level perceptron:")
    test_network_converges(mlp_test, 100)
    print("Testing a single convolutional layer:")
    test_network_converges(cl_test, 100)
    print("Testing small convolutional network:")
    test_network_converges(cnn_test, 100)
    """

def test_network_converges(network_test, networks):
    untrained = 0
    runaway = 0
    for i in range(networks):
        print("Network ", i + 1)
        test_mse = network_test()
        print("mse = ", test_mse)
        if test_mse > MSE_THRESHOLD:
            untrained += 1
        if math.isnan(test_mse):
            runaway += 1
    print("-----------------------------")
    print("Number of networks = ", networks)
    print("Trained (mse <", MSE_THRESHOLD, ") = ", networks - (untrained + runaway))
    print("Untrained (mse >", MSE_THRESHOLD, ") = ", untrained)
    print("Runaways (mse is NaN) = ", runaway)

def mlp_test():
    input_arr = FOUR_INPUTS
    ground_truth_arr = SUM_FOUR_INPUTS
    layer_sizes = [4, 1]
    mlp = Fully_Connected_Network(layer_sizes)
    train(mlp, input_arr, ground_truth_arr, 500)
    mse = test_network(mlp, input_arr, ground_truth_arr, False)
    return mse
def cl_test():
    training_set = BIN_GRID
    truth_set = BIN_GRID_CL_OUT
    cnn = Convolution_Layer(2, 1)
    train(cnn, training_set, truth_set, 30)
    network_mse = mse(cnn, training_set, truth_set)
    return network_mse.sum() / 4
def cnn_test():
    mlp = Fully_Connected_Network([4, 1])
    c_p = [Convolution_Layer(2, 1)]
    cnn = Convolutional_Network(c_p, mlp)

    training_set = BIN_GRID
    truth_set = BIN_GRID_CNN_OUT

    train(cnn, training_set, truth_set, 20)
    network_mse = mse(cnn, training_set, truth_set)
    return network_mse

def bees_train(epochs):
    cnn = bee_cnn()
    b_num = 0
    b_size = 5
    dataset = Dataset(.1)
    for epoch in range(epochs):
        for im_batch, point_batch in dataset.iter(True, b_size):
            b_num += 1
            print("batch", b_num)
            b_err = []
            for batch_index in range(len(im_batch)):

                img = im_batch[batch_index]
                rgb_inputs = to_ndlist(np.transpose(img))
                truth = len(point_batch[batch_index])

                prediction = cnn.forward(rgb_inputs)
                b_err.append(truth - prediction)
                cnn.backward(b_err[-1])
            print("mse = ", calc_mse(b_err))

def bee_cnn():
    c_p_layers = []
    c_p_layers.append([Pooling_Layer(3,3) for i in range(3)])
    c_p_layers.append([Convolution_Layer(5, 2) for i in range(3)])
    c_p_layers.append([Pooling_Layer(2,2) for i in range(3)])
    c_p_layers.append([Convolution_Layer(5, 2) for i in range(3)])
    c_p_layers.append([Pooling_Layer(2,2) for i in range(3)])
    c_p_layers.append([Convolution_Layer(5, 2) for i in range(3)])
    c_p_layers.append([Pooling_Layer(2,2) for i in range(3)])
    mlp = Fully_Connected_Network([12, 4, 2, 1])
    return Convolutional_Network(c_p_layers, mlp)

def train(network, dataset, truth, epochs):
    for i in range(epochs):
        for j in random.sample(range(len(dataset)), len(dataset)):
            out = network.forward(dataset[j])
            err = output_to_gradient(truth[j], out)
            network.backward(err)
    return dataset
def mse(network, dataset, truth):
    if isinstance(truth[0], Window):
        err = truth[0].copy().multiply(0)
        for i in range(len(dataset)):
            out = network.forward(dataset[i])
            grad = output_to_gradient(truth[i], out)
            err = err.add(grad.multiply(grad))
        return err.multiply(1 / len(dataset))
    err = 0
    for i in range(len(dataset)):
        out = network.forward(dataset[i])
        err += output_to_gradient(truth[i], out[0]) ** 2
    return err / len(dataset)
def test_network(mlp, input, ground_truth, show_test = False):
    squared_error_sum = 0
    for i in range(len(input)):
        prediction = mlp.forward(input[i])[0]
        actual = ground_truth[i]
        squared_error_sum += (prediction - actual) * (prediction - actual)
        if show_test:
            print("IPA: ", input[i], prediction, actual)
    return squared_error_sum / len(input)
def output_to_gradient(truth, out):
    if isinstance(out, list):
        if not isinstance(truth, list):
            truth = [truth]
        grad = []
        for i in range(len(out)):
            grad.append(output_to_gradient(truth[i], out[i]))
        return grad
    if isinstance(out, Window):
        return out.multiply(-1).add(truth)
    return truth - out
def to_ndlist(ndarr):
    copy = list(ndarr)
    if(isinstance(copy[0], numpy.ndarray)):
        copy = [to_ndlist(element) for element in copy]
    return copy
def calc_mse(err):
    sum = 0
    for element in err:
        sum += element **2
    return sum / len(err)

if __name__ == '__main__':
    main()
