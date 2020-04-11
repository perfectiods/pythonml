"""
via https://victorzhou.com/blog/intro-to-rnns/
Simple Sentiment Analysis task: determining whether a given text string is positive or negative.
Solved by Reccurent Neural Network (many-to-one type)
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





