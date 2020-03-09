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
        epochs = 1000 # number of cycles to loop throw dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * x[0] + self.w6 * x[1] + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                """
                Training of network.
                Lets use SGD (Stochastic Gradient) method, which says how we should change 
                weights and biases to make minimize losses.
                Yhe main equation is: w1 <- w1 - n * (dL/dw1)
                If dL/dw1 > 0, then w1 decreases, what causes that L1 decreases too.
                If dL/dw1 < 0, then w1 increases, what causes that L1 increases too.
                
                If we use this to every weight, losses will continuously decrease.
                
                What we do.
                1. Take one dataset.
                2. Calculate all derivatives by weight or  by bias.
                3. Use renovation equation for every weight or bias.
                4. Return to first point.
                """
                # Partial derivatives calculation
                d_L_d_y_pred = - 2 * (y_true - y_pred)

                # Neuron o1
                d_y_pred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_y_pred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_y_pred_d_b3 = deriv_sigmoid(sum_o1)

                d_y_pred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_y_pred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(h2)

                # Update weight and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_y_pred * d_y_pred_d_w5
                self.w6 -= learn_rate * d_L_d_y_pred * d_y_pred_d_w6
                self.b3 -= learn_rate * d_L_d_y_pred * d_y_pred_d_b3

                # Calculate total loss in the end of every phase
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print('epoch %d loss: %.3f' % (epoch, loss))


# Dataset definition
data = np.array([
    [-2, -1], #Alice
    [25, 6], #Bob
    [17, 4], #Cecil
    [-15, -6], #Diana
])

all_y_trues = np.array([
    1, #Alice
    0, #Bob
    0, #Cecil
    1, #Diana
])

# Train NN
network = NeuralNet()
network.train(data, all_y_trues)

# Make predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M