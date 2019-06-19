import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)