import numpy as np
class FFNN:
    def __init__(self, loss: str, n_hidden_layer: int, activation: str, batch: int, learning_rate: float) -> None:
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
