import numpy as np
# via https://python-scripts.com/intro-to-neural-networks

# #
# Let we have 1 neuron with 2 entries and 1 exit.
# Input.
# Let w1 = 0, w2 = 1. Or in vector form w = [0,1]; b = 4; x = [2,3]
#
# Inside
# Activation function f as sigmoid
# Neuron makes a scalar multiply: (w*x) + b = w1x1 + w2x2 + b = 0*2 + 1*3 + 4 = 7.
#
# Output
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

# Create neural network with feed forward propagation
class NeuralNetwork:
    """
    Neural network  with
    2 inputs,
    1 hidden layer with 2 neurons (h1,h2),
    1 output layer with 1 neuron (o1),
    every neuron has same weight and bias: w = [0,1], b = 0.
    """
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # Use class Neuron described previously
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
