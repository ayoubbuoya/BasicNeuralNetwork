
from numpy import *


class NeutralNetwork:
    def __init__(self, input_size=2, hidden_size=3, output_size=1, iters=1000):
        # parametres
        self.insize = input_size
        self.hidsize = hidden_size
        self.outsize = output_size
        self.iters = iters
        # Weights
        self.w1 = random.randn(self.insize, self.hidsize)
        self.w2 = random.randn(self.hidsize, self.outsize)

    def sig(self, z):
        return 1 / (1 + exp(-z))

    def sig_deriv(self, z):
        return z * (1 - z)

    def forward(self, x):
        self.z1 = dot(x, self.w1)
        self.a1 = self.sig(self.z1)
        self.z2 = self.a1
        self.z3 = dot(self.z2, self.w2)
        out = self.sig(self.z3)
        return out

    def backward(self, x, y, out):
        self.out_err = y - out
        self.out_delta = self.out_err * self.sig_deriv(out)

        self.z2_err = dot(self.z2.T, self.out_delta)
        self.z2_delta = self.z2_err * self.sig_deriv(self.z2)

        self.w1 += dot(x.T, self.z2_delta)
        self.w2 += dot(self.z2.T, self.out_delta)

    def fit(self, x, y):
        for _ in range(self.iters):
            out = self.forward(x)
            self.backward(x, y, out)

    def predict(self, x_pred):
        return self.forward(x_pred)


# tests
x = array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = array(([92], [86], [89]), dtype=float)
x_test = array(([4, 8]), dtype=float)
# transforme x to between 0 and 1 ==> divide every col / the max of col
x = x / amax(x, axis=0)
y = y / 100
x_test = x_test / amax(x_test, axis=0)

nn = NeutralNetwork()
nn.fit(x, y)
out = nn.predict(x_test)

print(out)
