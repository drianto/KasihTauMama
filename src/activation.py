# File untuk menyimpan perhitungan fungsi aktivasi
import numpy as np

def linear(x):
    return x

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hyperbolic_tangent(x):
    return np.tanh(x)

def softmax(x):
    return np.exp(x) / np.exp(x).sum()
