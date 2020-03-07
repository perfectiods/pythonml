import numpy as np
# via https://python-scripts.com/intro-to-neural-networks

# #
# Let we have 1 neuron with 2 entries and 1 exit.
# Input to neuron
# Let w1 = 0, w2 = 1. Or in vector form w = [0,1]; b = 4; x = [2,3]
#
# Inside the neuron
# Activation function f as sigmoid
# Neuron makes a scalar multiply: (w*x) + b = w1x1 + w2x2 + b = 0*2 + 1*3 + 4 = 7.
#
# Output from neuron
# Output function y = f(w1x1 + w2x2 + b); y = f(7) = 0,999.
##

def sigmoid(x):
    # Activation function f(x) = 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))

# Create one neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self, inputs):
        # Input parameters
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
weights = np.array([0,1]) # w1=0,w2=1
bias = 4 # b=4
n = Neuron(weights, bias)

x = np.array([2, 3]) # x 1=2,x2=3
print(n.feedforward(x)) # 0.9990889488055994


"""
Let we have some neurons. Lets collect them to Neural network.
Create Neural network with:
1 input layer with 2 neurons (x1, x2),
1 hidden layer with 2 neurons (h1,h2),
1 output layer with 1 neuron (o1),
every neuron has same weight and bias: w = [0,1], b = 0.

Calculation:
w = [0, 1], b = 0, x = [2, 3].
h1 = h2 = f(w*x + b) = f((0*2) + (1 * 3) + 0) = f(3) = 0,9526
o1 = f(w*[h1,h2] + b) = f((0*h1) + (1*h2) + 0) = f(0,9526) = 0,7216
"""

# Create described above neural network with feed forward propagation
class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # Use class Neuron described previously
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # Entries for o1 are outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.7216325609518421


