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
    Let L refer to cross-entropy loss: L = -ln(p_c) = -ln(softmax(y_c))
    
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
        n = len(self.last_inputs)

        # Calculate dL/dWhy and dL/dby
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Calculate dL/dh for the last h
        d_h = self.Why.T @ d_y

        # Back  propagate through time (BPTT)
        for t in reversed(range(n)):
            # An intermediate value: dL/dh * (1 - h^2)
            temp = temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.last_inputs[t].T

            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

            # Clip to prevent exploding gradients.
            for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
                np.clip(d, -1, 1, out=d)

            # Update weights and biases using gradient descent
            self.Whh -= learn_rate * d_Whh
            self.Wxh -= learn_rate * d_Wxh
            self.Why -= learn_rate * d_Why
            self.bh -= learn_rate * d_bh
            self.by -= learn_rate * d_by
            

