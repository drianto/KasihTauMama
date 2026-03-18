from __future__ import annotations
from activation import Activation
from loss import Loss

import numpy as np
import pickle
import matplotlib.pyplot as plt

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


class FFNN:
    def __init__(self, loss: Loss, layers: list[int], activation: Activation, link_verbose: bool, init_method: str, **kwargs) -> None:
        self.loss = loss
        self.link_verbose = link_verbose
        self.init_method = init_method
        self.init_kwargs = kwargs

        layer_count = len(layers)
        if layer_count < 2:
            raise ValueError(f"FFNN membutuhkan setidaknya 2 layer, layer count: {layer_count}")
        self.links = []
        for i in range(len(layers) - 1):
            link = Links(
                left_n_neuron=layers[i],
                right_n_neuron=layers[i + 1],
                activation_func=activation,
                verbose=link_verbose,
                init_method=init_method,
                **kwargs
            )
            self.links.append(link)


    def add_link(self, position: int, n_neuron: int, activation: Activation) -> None:
        """Menambah layer pada indeks posisi tertentu."""
        if position < 0 or position > len(self.links):
            raise IndexError("Posisi di luar jangkauan")
        
        left_n = self.links[0].weight.shape[0] if position == 0 else self.links[position - 1].weight.shape[1]
        right_n = self.links[-1].weight.shape[1] if position == len(self.links) else self.links[position].weight.shape[0]

        new_link = Links(left_n, n_neuron, activation, self.link_verbose, self.init_method, **self.init_kwargs)

        if position < len(self.links):
            old_link = self.links[position]
            adjusted = Links(n_neuron, right_n, old_link.activation, self.link_verbose, self.init_method, **self.init_kwargs)
            self.links[position] = adjusted

        self.links.insert(position, new_link)


    def remove_link(self, position: int) -> None:
        """Menghapus layer pada indeks posisi tertentu."""
        if len(self.links) <= 1:
            raise ValueError("Tidak bisa menghapus link, FFNN minimal 1 link.")
        if position < 0 or position >= len(self.links):
            raise IndexError("Posisi di luar jangkauan")

        self.links.pop(position)
        if position < len(self.links):
            left_n = self.links[position - 1].weight.shape[1] if position > 0 else self.links[0].weight.shape[0]
            right_n = self.links[position].weight.shape[1]
            old_act = self.links[position].activation
            adjusted = Links(left_n, right_n, old_act, self.link_verbose, self.init_method, **self.init_kwargs)
            self.links[position] = adjusted


    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, batch_size: int, verbose=1, X_val=None, y_val=None, l1=0.0, l2=0.0) -> dict:
        if X.shape[1] != self.links[0].weight.shape[0]:
            raise ValueError(
                f"Jumlah fitur: {X.shape[1]} tidak sesuai dengan jumlah neuron layer pertama: {self.links[0].weight.shape[0]}"
            )

        if y.shape[1] != self.links[-1].weight.shape[1]:
            raise ValueError(
                f"Jumlah target fitur: {y.shape[1]} tidak sesuai dengan jumlah neuron output layer terakhir: {self.links[-1].weight.shape[1]}"
            )

        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        for epoch in range(epochs):
            loss_list = []
            for i in range(0, len(X), batch_size):
                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                y_pred = self.forward(xb)
                batch_loss = self.loss.forward(y_pred, yb)
                loss_list.append(batch_loss)

                self.backward(y_pred, yb)
                self.update_weight(learning_rate, l1, l2)

                if verbose == 1:
                    progress = (b_idx + 1) / total_batches
                    bar_len = 30
                    filled_len = int(bar_len * progress)
                    bar = '=' * filled_len + '.' * (bar_len - filled_len)
                    sys.stdout.write(f"\rEpoch {epoch + 1}/{epochs} [{bar}] {int(progress * 100)}%")
                    sys.stdout.flush()

            train_loss = np.mean(loss_list)
            history["train_loss"].append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss.forward(y_val_pred, y_val)
                history["val_loss"].append(val_loss)

            if verbose == 1:
                msg = f" - loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                print(msg)
        
        return history


    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(path) -> FFNN:
        with open(path, "rb") as f:
            return pickle.load(f)


    def forward(self, x):
        for link in self.links:
            x = link.forward(x)

        return x


    def backward(self, y_pred, y_true):
        gradient = self.loss.backward(y_pred, y_true)
        for link in reversed(self.links):
            gradient = link.backward(gradient)


    def update_weight(self, learning_rate: float, l1=0, l2=0):
        for link in self.links:
            self.apply_regularization(link, l1, l2)
            link.weight -= learning_rate * link.dw
            link.bias -= learning_rate * link.db


    def apply_regularization(self, layer, l1=0, l2=0):
        if l1 > 0:
            layer.dw += l1 * np.sign(layer.weight)

        if l2 > 0:
            layer.dw += 2 * l2 * layer.weight


    def show_weight_distribution(self, layers: list[int]) -> None:
        """Plot distribusi bobot dan bias pada layer tertentu."""
        fig, axes = plt.subplots(len(layers), 2, figsize=(10, 4 * len(layers)))
        if len(layers) == 1: axes = np.array([axes])

        for idx, (ax_w, ax_b), l_idx in zip(range(len(layers)), axes, layers):
            if 0 <= l_idx < len(self.links):
                ax_w.hist(self.links[l_idx].weight.flatten(), bins=30, color='blue', alpha=0.7)
                ax_w.set_title(f"Layer {l_idx} Weight")
                ax_b.hist(self.links[l_idx].bias.flatten(), bins=30, color='red', alpha=0.7)
                ax_b.set_title(f"Layer {l_idx} Bias")
        plt.tight_layout()
        plt.show()


    def show_dw_distribution(self, layers: list[int]) -> None:
        """Plot distribusi gradien bobot dan bias pada layer tertentu."""
        fig, axes = plt.subplots(len(layers), 2, figsize=(10, 4 * len(layers)))
        if len(layers) == 1: axes = np.array([axes])

        for idx, (ax_w, ax_b), l_idx in zip(range(len(layers)), axes, layers):
            if 0 <= l_idx < len(self.links):
                if self.links[l_idx].dw is not None:
                    ax_w.hist(self.links[l_idx].dw.flatten(), bins=30, color='green', alpha=0.7)
                ax_w.set_title(f"Layer {l_idx} dW")
                if self.links[l_idx].db is not None:
                    ax_b.hist(self.links[l_idx].db.flatten(), bins=30, color='orange', alpha=0.7)
                ax_b.set_title(f"Layer {l_idx} dB")
        plt.tight_layout()
        plt.show()


def main():
    print("Hello from kasihtaumama!")

    class DummyActivation(Activation):
        def forward(self, x): return x
        def backward(self, x): return np.ones_like(x)

    class DummyLoss(Loss):
        def forward(self, y_pred, y_true):
            return np.mean((y_pred - y_true)**2)
        def backward(self, y_pred, y_true):
            return 2 * (y_pred - y_true) / y_true.shape[0]

    # Dummy dataset
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y = np.random.rand(100, 1)  # 100 targets, 1 output

    model = FFNN(
        loss=DummyLoss(),
        layers=[3, 5, 4, 1],
        activation=DummyActivation(),
        link_verbose=False,
        init_method='uniform',
        lower=-0.1,
        upper=0.1,
        seed=42
    )

    # Train
    history = model.fit(X, y, epochs=5, learning_rate=0.01, batch_size=10, verbose=1, l1=0.001, l2=0.001)

    model.show_weight_distribution(layers=[0, 1])
    model.show_dw_distribution(layers=[0, 1])

    model.add_link(1, 10, DummyActivation())
    model.remove_link(1)

    # Predict
    y_pred = model.predict(X[:5])
    print("Sample predictions:\n", y_pred)

    # Save and Load
    # model.save("test_model.pkl")
    # loaded_model = FFNN.load("test_model.pkl")
    # y_pred_loaded = loaded_model.predict(X[:5])
    # print("Predictions from loaded model:\n", y_pred_loaded)


if __name__ == "__main__":
    main()
