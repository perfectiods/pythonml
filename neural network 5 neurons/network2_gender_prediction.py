import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true, y_pred - Numpy arrays
    return((y_true - y_pred)**2).mean()

class NeuralNet:
    """
    Neural Network with:
    1 input layer with 2 neurons (x1, x2),
    1 hidden layer with 2 neurons (h1,h2),
    1 output layer with 1 neuron (o1),
    """
    def __init__(self):
        # Weight
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x = [0, 1] - entry
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        # n - number of samples in dataset, all_y_trues - np array with n elements
        learn_rate = 0.1
        epochs = 1000 # number of cycles in dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * x[0] + self.w6 * x[1] + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # Derivatives calculation
                # 1 2 3
