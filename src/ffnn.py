from typing import Callable
import numpy as np

class Links:
    def __init__(self) -> None:
        self.weight: np.ndarray =
        self.bias: np.ndarray
        


class FFNN:
    def __init__(self, loss: Callable, n_hidden_layer: int, activation: Callable, batch: int, learning_rate: float) -> None:
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
