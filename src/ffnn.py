class FFNN:
    def __init__(self, loss, n_hidden_layer, activation, batch, learning_rate):
        self.loss = loss
        self.n_hidden_layer = n_hidden_layer
        self.activation = activation
        self.batch = batch
        self.learning_rate = learning_rate

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

def main():
    print("Hello from kasihtaumama!")


if __name__ == "__main__":
    main()
