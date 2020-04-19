import numpy as np
from numpy.random import randn


class RNN:

    # Vanilla neural network
    def __init__(self, input_size, output_size, hidden_size=64):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        Perform a forward pass to RNN
        Return final output and hidden state
        Inputs are one-hot vectors with shape (input_size, 1)
        """
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}


        # Perform each step of the RNN
        for i, x in enumerate(inputs): # enumerate - makes array like (index, arrays value)
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

            self.last_hs[i + 1] = h

        # Calculate output:
        y = self.Why @ h + self.by
        return y, h

    """
    Let y represent RNN's output (raw)
    Let p represent final probs: p = softmax(y)
    Let c refer to "correct" class
    
    In order to train RNN we need a loss function. Lets use cross-entropy L = -ln(p_c),
    where p_c - our RNN's predicted probability for correct class.
    So after we have loss we can train RNN to minimize it. We'll use gradient descent.
    
    """
    def backprop(self, d_y, learn_rate=2e-2):
        """
        Perform backprop of RNN
        - d_y = dL/dy has shape (output_size, 1)
        - learn rate is float
        """
        pass





