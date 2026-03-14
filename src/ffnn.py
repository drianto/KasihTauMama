from activation import Activation
from loss import Loss

import numpy as np
import pickle

class Links:
    def __init__(
            self,
            left_n_neuron: int,
            right_n_neuron: int,
            activation_func: Activation,
            verbose: bool,
            init_method: str, # 'zero', 'uniform', 'normal'
            **kwargs # arguments buat init method
            # zero: no argument
            # uniform: lower, upper, seed
            # normal: mean, variance, seed
        ) -> None:
        self.activation = activation_func
        self.left_layer = None
        self.weighted_sum = None # type: ignore # nilai weighted_sum sebelum masuk ke fungsi aktivasi
        self.dw = None
        self.db = None
        self.verbose = verbose

        if init_method.lower() == 'zero':
            self.weight = np.zeros((left_n_neuron, right_n_neuron))
            self.bias = np.zeros((right_n_neuron,))
            if self.verbose:
                with np.printoptions(precision=4, suppress=True, linewidth=100):
                    print(f"Weight:\n{self.weight}")
                    print(f"Bias:\n{self.bias}")
        elif init_method.lower() == 'uniform':
            lower = kwargs.get('lower', -0.5)
            upper = kwargs.get('upper', 0.5)
            seed = kwargs.get('seed', None)
            if seed is not None:
                np.random.seed(seed)

            self.weight = np.random.uniform(lower, upper, (left_n_neuron, right_n_neuron))
            self.bias = np.random.uniform(lower, upper, (right_n_neuron,))

            if self.verbose:
                with np.printoptions(precision=4, suppress=True, linewidth=100):
                    print(f"Weight:\n{self.weight}")
                    print(f"Bias:\n{self.bias}")
        elif init_method.lower() == 'normal':
            mean = kwargs.get('mean', 0.0)
            variance = kwargs.get('variance', 1.0)
            seed = kwargs.get('seed', None)
            if seed is not None:
                np.random.seed(seed)

            self.weight = np.random.normal(mean, np.sqrt(variance), (left_n_neuron, right_n_neuron))
            self.bias = np.random.normal(mean, np.sqrt(variance), (right_n_neuron,))

            if self.verbose:
                with np.printoptions(precision=4, suppress=True, linewidth=100):
                    print(f"Weight:\n{self.weight}")
                    print(f"Bias:\n{self.bias}")
        else:
            raise ValueError(f"Init method tidak diketahui: {init_method}")


    def forward(self, X: np.ndarray) -> np.ndarray:
        self.left_layer = X
        self.weighted_sum = X @ self.weight + self.bias
        return self.activation.forward(self.weighted_sum)


    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        if self.weighted_sum is None:
            raise ValueError("Forward propagation dilakukan, weight is None")
        if self.left_layer is None:
            raise ValueError("Forward propagation belum dilakukan, left layer is None")

        gradient_activation = gradient_output * self.activation.backward(self.weighted_sum)
        batch_size = self.left_layer.shape[0]
        self.dw = self.left_layer.T @ gradient_activation / batch_size
        self.db = np.sum(gradient_activation, axis=0) / batch_size
        return gradient_activation @ self.weight.T


    def show_weight_distribution(self) -> None:
        pass


    def show_dw_distribution(self) -> None:
        pass


class FFNN:
    def __init__(self, loss: Loss, n_hidden_layer: int, activation: Activation, batch: int, learning_rate: float) -> None:
        self.loss = loss
        self.n_hidden_layer = n_hidden_layer
        self.activation = activation
        self.batch = batch
        self.learning_rate = learning_rate


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(path) -> FFNN:
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    print("Hello from kasihtaumama!")


if __name__ == "__main__":
    main()
