import random
import math
import numpy
from cnn.window import *


LEARNING_RATE = 0.1
MOMENTUM_FACTOR = 0
LEAK = 0
BN_SMOOTH = 1

def random_weight(n):
    return random.random() * 2 - 1
    #return numpy.random.randn() * math.sqrt(2.0 /n)
def mse_grad(expected, outputs):
    if(len(expected) != len(outputs)):
        raise Exception("expected and output array sizes don't match")
    result = []
    size = len(expected)
    for i in range(size):
        -2 * result.append(expected[i] - outputs[i]) / size
    return result
def dot(arr1, arr2):
    arr1 = arr_to_2d(arr1)
    arr2 = arr_to_2d(arr2)
    if len(arr1) != len(arr2[0]):
        raise Exception("Wrong array sizes for dot multiplication")
    result = []
    for i in range(len(arr2)):
        column = []
        for j in range(len(arr1[0])):
            sum = 0
            for k in range(len(arr2[i])):
                sum += arr1[k][j] * arr2[i][k]
            column.append(sum)
        result.append(column)
    return reduce_dimension(result)
def arrT(arr):
    arr = arr_to_2d(arr)
    result = []
    for i in range(len(arr[0])):
        result.append([])
        for j in range(len(arr)):
            result[i].append(arr[j][i])
    return reduce_dimension(result)
def arr_to_2d(arr):
    if not isinstance(arr, list):
        arr = [arr]
    if not isinstance(arr[0], list):
        temp = []
        for i in range(len(arr)):
            temp.append([arr[i]])
        arr = temp
    return arr
def reduce_dimension(arr):
    result = []
    for i in range(len(arr)):
        if not isinstance(arr[i], list) or len(arr[i]) > 1:
            return arr
        result.append(arr[i][0])
    return result
def activation_function(x):
    return ReLU(x)
def derived_activation(x):
    return derived_ReLU(x)
def ReLU(x):
    if isinstance(x, list):
        result = []
        for i in range(len(x)):
            result.append(ReLU(x[i]))
        return result
    else:
        if x >= 0:
            return x
        return LEAK * x
def derived_ReLU(x):
    if isinstance(x, list):
        result = []
        for i in range(len(x)):
            result.append(derived_ReLU(x[i]))
        return result
    else:
        if x < 0:
            return LEAK
        return 1
def sigmoid(x):
    if isinstance(x, list):
        result = []
        for i in range(len(x)):
            result.append(sigmoid(x[i]))
        return result
    return 1 / (1 + math.e ** -x)
def derived_sigmoid(x):
    if isinstance(x, list):
        result = []
        for i in range(len(x)):
            result.append(derived_sigmoid(x[i]))
        return result
    return sigmoid(x) * (1 - sigmoid(x))
def normalize(x, rescale, shift):
    return x#batch_normalization(x, rescale, shift)
def der_normalize(x, gradients, rescale):
    return x#derived_batch_norm(x, gradients, rescale)
def batch_normalization(x, rescale, shift):
    return op(op(get_x_hat(x), rescale, "*"), shift, "+")
def stdev(x):
    avg = sum(x) / len(x)
    sqrdif = op(op(x, avg, "-"), 2, "**")
    return math.sqrt(sum(sqrdif) / len(x))
def get_x_hat(x):
    avg_x = sum(x) / len(x)
    stdev_x = stdev(x)
    return op(op(x, avg_x, "-"), math.sqrt((stdev_x ** 2) + BN_SMOOTH), "/")
def derived_batch_norm(x, gradients, rescale):
    avg = sum(x) / len(x)
    dif = op(x, avg, "-")
    s = 1 / (stdev(x)**2 + BN_SMOOTH)
    dx_hat = op(gradients, rescale, "*")
    davg = -1 * math.sqrt(s) * sum(dx_hat)
    dsqr_stdev = op((-1/2) * (s ** 1.5) * sum(dx_hat), dif, "*")
    left = op(dx_hat, math.sqrt(s), "*")
    right = op(op(op(dsqr_stdev, 2, "*"), dif, "*"), davg, "+")
    res = op(left, op(right, 1/len(x), "*"), "+")
    return res
def sum_columns(arr):
    if not isinstance(arr, list) or not isinstance(arr[0], list):
        return arr
    result = []
    for column in arr:
        result.append(sum(column))
    return result
def op(arr1, arr2, sign):
    if isinstance(arr1, list) and isinstance(arr2, list):
        if len(arr1) != len(arr2):
            raise Exception("Size mismatch")
        result = []
        for i in range(len(arr1)):
            result.append(op(arr1[i], arr2[i], sign))
        return result
    elif isinstance(arr1, list):
        result = []
        for i in range(len(arr1)):
            result.append(op(arr1[i], arr2, sign))
        return result
    elif isinstance(arr2, list):
        result = []
        for i in range(len(arr2)):
            result.append(op(arr1, arr2[i], sign))
        return result
    match(sign):
        case "*":
            return arr1 * arr2
        case "/":
            return arr1 / arr2
        case "-":
            return arr1 - arr2
        case "+":
            return arr1 + arr2
        case "**":
            return arr1 ** arr2
        case _:
            raise Exception("Invalid sign")

def generate_bin_squares(side_len):
    training_set = []
    for i in range(2**(side_len**2)):
        temp = []
        for j in range(side_len ** 2):
            if math.floor(i / (2 ** j)) % 2 == 1:
                temp.append(1)
            else:
                temp.append(0)
        training_set.append(unflatten(temp))
    return training_set
def filter_features(set, filter):
    truth_set = []
    for input in set:
        truth = []
        for i in range(input.size - (filter.size - 1)):
            temp = []
            for j in range(input.size - (filter.size - 1)):
                if input.get_section(i, j, filter.size) == filter:
                    temp.append(1)
                else:
                    temp.append(0)
            truth.append(temp)
        truth_set.append(Window(truth))
    return truth_set
def count_features(set, filter):
    truth_set = filter_features(set, filter)
    for i in range(len(truth_set)):
        truth_set[i] = truth_set[i].sum()
    return truth_set