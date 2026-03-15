# File untuk menghitung loss function
class Loss:
    # Fungsi loss biasa buat forward prop
    def forward(self, y_pred, y_true):
        raise NotImplementedError


    # Fungsi loss turunan buat backward prop
    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, y_pred, y_true):
        raise NotImplementedError


    def backward(self, y_pred, y_true):
        raise NotImplementedError


class BCE(Loss):
    def forward(self, y_pred, y_true):
        raise NotImplementedError


    def backward(self, y_pred, y_true):
        raise NotImplementedError


class CCE(Loss):
    def forward(self, y_pred, y_true):
        raise NotImplementedError


    def backward(self, y_pred, y_true):
        raise NotImplementedError
