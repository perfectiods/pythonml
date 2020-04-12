"""
via https://victorzhou.com/blog/intro-to-rnns/
Simple Sentiment Analysis task: determining whether a given text string is positive or negative.
Solved by Recurrent Neural Network (many-to-one type)
"""

from data import test_data, train_data

# Create the vocabulary
vocab = list(set(
    [w for text in train_data.keys() for w in text.split(' ')]
))

vocab_size = len(vocab)
print('%d unique words found' %vocab_size) # 18 unique words found

# Assign indexes to each word
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
print(word_to_idx['good'])
print(idx_to_word[0])

# Use one-hot vectors,
# which contain all zeros except for a single one. The “one” in each one-hot vector will
# be at the word’s corresponding integer index

inputs = []

for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
return inputs

# FORWARD PHASE
# Initialize 3 weights and 2 biases

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





