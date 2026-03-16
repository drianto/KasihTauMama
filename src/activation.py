# File untuk menyimpan perhitungan fungsi aktivasi
import numpy as np

class Activation:
    # Fungsi aktivasi biasa buat forward prop
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    # Fungsi aktivasi turunan buat backward prop
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (np.ones_like(grad))


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.val = x
        return np.maximum(0, x)


    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(self.val > 0, grad, 0)

class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.val = 1 / (1 + np.exp(-1 * x))
        return self.val


    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.val * (1 - self.val))


class HyperbolicTangent(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.val = x
        return np.tanh(x)


    def backward(self, grad: np.ndarray) -> np.ndarray:
        return  grad * (2 / (np.exp(self.val) - np.exp(-self.val))) ** 2


class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exponents = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.vals = exponents / np.sum(exponents, axis=-1, keepdims=True)
        return self.vals


    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size = self.vals.shape[0]
        dx = np.empty_like(self.vals)

        for i in range (batch_size):
            si = self.vals[i].reshape(-1, 1)
            calc = np.diagflat(si) - np.dot(si, si.T)
            dx[i] = np.dot(calc, grad[i])

        return dx
