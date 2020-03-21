"""
via https://victorzhou.com/blog/keras-neural-network-tutorial/
Classic ML problem: MNIST handwritten digit classification. Given an image, classify it as digit.
Solved by using Keras library on TensorFlow backend.
"""

"""
Note: every run of this script builds and trains model. We don't need to do it every time.
Instead we run this code for one time to prepare data, build model, test and use it.
So after that steps we obtain trained model.
So we can start making predictions using it.

How to do that.
In Step 2 we use save_weights method to save trained model output to harddisk. This is done in first run of script.
So to start making predictions using a model we should only run Step 2.1 and Step 4 with load_weights method uncommented.
"""

import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical



# 1. Prepare data
# Download and cache the data (photos)
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape) #(60000,28,28)
print(train_labels.shape) #(60000)

# Normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape) #(60000,784)
print(train_labels.shape) #(10000,784)


# 2. Build the model
# 2.1 Every Keras model is built using Sequential class which represents a linear stack of layers
model = Sequential([
    # layers:
    Dense(64, activation='relu', input_shape=(784,)), # layer has 64 nodes, each use ReLU activation f-n
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'), # output layer has 10 nodes, each has Softmax act-n f-n
])

# Compile the model
# We decide 3 factors: the optimizer, the loss function, a list of metrics
model.compile(
    optimizer='adam', #Adam - an algorithm for first-order gradient-based optimization of stochastic objective functions
    loss='categorical_crossentropy', #Loss f-n which is used as feedback for weight tensors training. Seeks to reduce.
    metrics=['accuracy'],
)

# Train the model
# We decide 3 parameters: training data, number of epochs, batch size

model.fit(
    train_images, #training data: (images and labels), commonly known as X and Y, respectively.
    to_categorical(train_labels), #represent decimal into 10 dimensional array, i.e. 2 = [0,0,1,0,0,0,0,0,0]
    epochs=5, #iterations over the entire dataset
    batch_size=32, #number of samples per gradient update
)
# May run on this point. After 5 epochs: 11s 188us/step - loss: 0.1077 - accuracy: 0.9662
# Training a model in Keras literally consists only of calling fit() and specifying some parameters.


# 3. Test the model
# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels),
)
# Gives 0s 15us/step [0.10821614159140736, 0.965] - that means that evaluate() method returned loss array
# Loss: 0.108, accuracy: 96.5%.


# 4. Use the model
# Save model to disk
model.save_weights('model.h5') #comment this after first run

# We can now reload the trained model whenever we want by rebuilding it (step 2. Build)
# and loading in the saved weights:
#model.load_weights('model.h5') # uncomment this after first run

# So when we reload model we can now make predictions
# Note: because of Softmax we have 10 probabilities. To turn them into digits use np.argmax()method
# which returns index of arrays maximum element

# Predict on the first 5 images:
predictions = model.predict(test_images[:5])

# Print model predictions
print(np.argmax(predictions, axis=1))

# Test our predictions against ground truth
print(test_labels[:5])

"""
Tuning variants
1. Increase epochs and batch size. Effect: loss decreases, accuracy increases [good]
Fit with epochs = 5 and batch_size = 32 gives:
60000/60000 [==============================] - 9s 156us/step - loss: 0.1073 - accuracy: 0.9669 [first try]
Fit with epochs = 10 and batch_size = 64 gives:
60000/60000 [==============================] - 6s 96us/step - loss: 0.0707 - accuracy: 0.9767 [become better]

2. Add 3 more layers with ReLu act. f-n. Effect: loss increases, accuracy decreases [bad]
Fit with epochs = 5, batch_size = 32 gives, 5 layers with ReLu gives:
14s 228us/step - loss: 0.1165 - accuracy: 0.9643 [become worst]




"""

"""
Use this to make predictions with trained model
"""

"""
model = Sequential([
    # layers:
    Dense(64, activation='relu', input_shape=(784,)), # layer has 64 nodes, each use ReLU activation f-n
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'), # output layer has 10 nodes, each has Softmax act-n f-n
])

model.load_weights('model.h5')

# Predict on the first 5 images:
predictions = model.predict(test_images[:5])

# Print model predictions
print(np.argmax(predictions, axis=1))

# Test our predictions against ground truth
print(test_labels[:5])

"""






