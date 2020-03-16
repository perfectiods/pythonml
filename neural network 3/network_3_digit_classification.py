# via https://victorzhou.com/blog/keras-neural-network-tutorial/
# Classic ML problem: MNIST handwritten digit classification. Given an image, classify it as digit.

import numpy as np
import mnist
import keras

# 1. Prepare data
# download and cache the data (photos)
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape) #(60000,28,28)
print(train_labels.shape) #(60000)

# normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# flatten the images
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape) #(60000,784)
print(train_labels.shape) #(10000,784)

# 2. Build the model
# Every Keras model is built using Sequential class which represents a linear stack of layers

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    # layers:
    Dense(64, activation='relu', input_shape=(784,)), # layer has 64 nodes, each use ReLU activation f-n
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'), # output layer has 10 nodes, each has Softmax act-n f-n
])

# compile the model
# we decide 3 factors: the optimizer, the loss function, a list of metrics

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics='accuracy',
)

# train the model
# we decide 3 parameters: trainig data, number of epochs, batch size

model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=32,
)

