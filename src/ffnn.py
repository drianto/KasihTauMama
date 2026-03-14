from activation import Activation

import numpy as np

class Links:
    def __init__(
            self,
            left_n_neuron: int,
            right_n_neuron: int,
            activation_func: Activation,
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

        if init_method.lower() == 'zero':
            self.weight = np.zeros((left_n_neuron, right_n_neuron))
            self.bias = np.zeros((right_n_neuron,))
        elif init_method.lower() == 'uniform':
            lower = kwargs.get('lower', -0.5)
            upper = kwargs.get('upper', 0.5)
            seed = kwargs.get('seed', None)
            if seed is not None:
                np.random.seed(seed)

            self.weight = np.random.uniform(lower, upper, (left_n_neuron, right_n_neuron))
            self.bias = np.random.uniform(lower, upper, (right_n_neuron,))
        elif init_method.lower() == 'normal':
            mean = kwargs.get('mean', 0.0)
            variance = kwargs.get('variance', 1.0)
            seed = kwargs.get('seed', None)
            if seed is not None:
                np.random.seed(seed)

            self.weight = np.random.normal(mean, np.sqrt(variance), (left_n_neuron, right_n_neuron))
            self.bias = np.random.normal(mean, np.sqrt(variance), (right_n_neuron,))
        else:
            raise ValueError(f"Init method tidak diketahui: {init_method}")


    def forward(self, X: np.ndarray) -> np.ndarray:
        pass


    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        pass


class FFNN:
    def __init__(self, loss, n_hidden_layer: int, activation: Activation, batch: int, learning_rate: float) -> None:
        self.loss = loss
        self.n_hidden_layer = n_hidden_layer
        self.activation = activation
        self.batch = batch
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

def main():
    print("Hello from kasihtaumama!")


if __name__ == "__main__":
    main()
