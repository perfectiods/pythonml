"""
via https://victorzhou.com/blog/intro-to-rnns/
Simple Sentiment Analysis task: determining whether a given text string is positive or negative.
Solved by Recurrent Neural Network (many-to-one type)
"""

import numpy as np
from data import test_data, train_data
from network_4_rnn import RNN

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

def createInputs(text):
    """
    Return an array of one-hot vectors representing each word
    - text is a string
    - each one-hot vector has shape (vocab_size, 1)
    """
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs

def softmax(xs):
    # Apply Softmax Function to input array
    return np.exp(xs) / sum(np.exp(xs))


# Initialize RNN
rnn = RNN(vocab_size, 2)

"""
To test RNN working try to forward pass and count probabilities
"""
inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs)


# FORWARD PHASE
# Initialize 3 weights and 2 biases
