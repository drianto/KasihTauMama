# File untuk menyimpan perhitungan fungsi aktivasi
import numpy as np

class Activation:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Linear(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class HyperbolicTangent(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
