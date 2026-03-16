# File untuk menghitung loss function
import numpy as np

eps = 1e-15 # konstanta buat ngehinadari log(0)  
class Loss:
    # Fungsi loss biasa buat forward prop
    def forward(self, y_pred, y_true):
        raise NotImplementedError


    # Fungsi loss turunan buat backward prop
    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, y_pred, y_true):
        sum = np.power(y_true - y_pred, 2)
        return np.mean(sum)

    def backward(self, y_pred, y_true):
        n = y_true.size
        return -2 * (y_true - y_pred) / n


class BCE(Loss):
    def forward(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


    def backward(self, y_pred, y_true):
        n = y_true.size
        return (-1) * (y_true - y_pred) / (y_pred * (1 - y_pred) + eps) / n


class CCE(Loss):
    def forward(self, y_pred, y_true):
        n = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + eps)) / n


    def backward(self, y_pred, y_true):
        n = y_true.shape[0]
        return -(y_true / (y_pred + eps)) / n
