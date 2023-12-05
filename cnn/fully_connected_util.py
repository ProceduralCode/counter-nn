import random
import math

LEARNING_RATE = 0.5
MOMENTUM_FACTOR = 0.1

def random_weight():
    return random.random()
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
    return sigmoid(x)
def derived_activation(x):
    return derived_sigmoid(x)
def ReLU(x):
    if isinstance(x, list):
        result = []
        for i in range(len(x)):
            result.append(ReLU(x[i]))
        return result
    else:
        return max(0, x)
def derived_ReLU(x):
    if isinstance(x, list):
        result = []
        for i in range(len(x)):
            result.append(derived_ReLU(x[i]))
        return result
    else:
        if(x < 0):
            return 0
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
    return x * (1 - x)