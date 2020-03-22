"""
via https://victorzhou.com/blog/keras-cnn-tutorial/#the-full-code
Classic ML problem: MNIST handwritten digit classification. Given an image, classify it as digit.
Solved by using Keras and Convolutional Neyral Network.
"""

"""
What are Convolutional Neural Networks?

They’re basically just neural networks that use Convolutional layers,
which are based on the mathematical operation of convolution.
Conv layers consist of a set of filters - 2d matrices of numbers.

Convolution - свёртка. Используется фильтр - матрица 3х3 накладывается на матрицу исх. изображения 4х4.
Матрица исх. изобр. состоит из кодов цветов GRB от 0 до 255.
Фильтр 4 раза суммируется с весами. Получается матрица свёртки 2х2. Это - выходное изображение.
"""

import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
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

# Reshape the images
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 8
filter_size = 3 #filter is matrix 3x3
pool_size = 2 #traverse the input image in 2x2 blocks


"""
# 2. Build the model
model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)), #input layer
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'), #output softmax layer has 10 nodes
])

# Compile the model
# We decide 3 factors: the optimizer, the loss function, a list of metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
# We decide 3 parameters: training data, number of epochs, batch size
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32,
)
# Run on this point gives: 13s 211us/step - loss: 0.0866 - accuracy: 0.9747

# 3. Test the model
# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels),
)
# Gives 18s 293us/step - loss: 0.0909 - accuracy: 0.9736 - that means that evaluate() method returned loss array.
# Loss: 0.909, accuracy: 97.36%.


# 4. Use the model
# Save model to disk
model.save_weights('model.h6') #comment this after first run

# We can now reload the trained model whenever we want by rebuilding it (step 2. Build)
# and loading in the saved weights:
#model.load_weights('model.h5') # uncomment this after first run

# Predict on the first 5 images:
predictions = model.predict(test_images[:5])

# Print model predictions
print(np.argmax(predictions, axis=1))

# Test our predictions against ground truth
print(test_labels[:5])
"""


"""
Use this to make predictions with trained model
"""


model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)), #input layer
    MaxPooling2D(pool_size=pool_size), #pooling - decrease redundance. We take only max value from 4 neighbor pixels
    Flatten(),
    Dense(10, activation='softmax'), #output softmax layer has 10 nodes
])

model.load_weights('model.h6')

# Predict on the first 5 images:
predictions = model.predict(test_images[:5])

# Print model predictions
print(np.argmax(predictions, axis=1))

# Test our predictions against ground truth
print(test_labels[:5])




