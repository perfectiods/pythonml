import numpy as np
from numpy.random import randn

class RNN:

    # Vanilla neural network
    def __init__(self, input_size, output_size, hidden_size = 64):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000
        # Biases
        self.bh = np.zeros(hidden_size,1)
        self.by = np.zeros(hidden_size, 1)

    def forward(self, inputs):
        """
        Perform a forward pass to RNN
        Return final output and hidden state
        Inputs are one-hot vectors with shape (input_size, 1)
        """
        h = np.zeros(self.Whh.shape[0], 1)
        # Perform each step of the RNN
        for i,x in enumerate(inputs): # enumerate - makes array like (index, arrays value)
            h = np.tanh(self.Wxh @ x + self.Whh @ x + self.bh)
        # Calculate output:
        y = self.Why @ h + self.by
    return y, h







