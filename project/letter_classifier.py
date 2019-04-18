#!/usr/bin/python3

from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pnslib import utils
import cv2
import string

import os
from tensorflow import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity(logging.ERROR)

# Load all the ten classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels

X = []
y = []

letters = string.ascii_uppercase

for image in os.listdir("data/letters/"):
  im = cv2.imread("data/letters/" + image)
  letter = image.split(".")[0].upper()

  index = letters.index(letter)

  print(letter, index)
  X.append(im)
  y.append(index)

train_x = np.array(X)
train_y = np.array(y)

test_x = np.array(X)
test_y = np.array(y)

"""
(train_x, train_y, test_x, test_y) = utils.fashion_mnist_load(
    data_type="full", flatten=False)
"""

print(train_y)

num_classes = len(string.ascii_uppercase)

print("[MESSAGE] Dataset is loaded.")

# preprocessing for training and testing images
"""
train_x = train_x.astype("float32") / 255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32") / 255.
test_x -= mean_train_x
"""

print("[MESSAGE] Dataset is preprocessed.")

print(test_x.shape)

# converting the input class labels to categorical labels for training
train_Y = to_categorical(train_y, num_classes=num_classes)
test_Y = to_categorical(test_y, num_classes=num_classes)

print("[MESSAGE] Converted labels to categorical labels.")

# define a model
num_train_samples = train_x.shape[0]
num_test_samples = test_x.shape[0]
input_shape = train_x.shape[1:]

kernel_sizes = [(7, 7), (5, 5)]
num_kernels = [20, 25]

pool_sizes = [(2, 2), (2, 2)]
pool_strides = [(2, 2), (2, 2)]

num_hidden_units = 200

x = Input(shape=input_shape)
y = Conv2D(num_kernels[0], kernel_sizes[0], activation='relu')(x)
y = MaxPooling2D(pool_sizes[0], pool_strides[0])(y)
y = Conv2D(num_kernels[1], kernel_sizes[1], activation='relu')(y)
y = MaxPooling2D(pool_sizes[1], pool_strides[1])(y)
y = Flatten()(y)
y = Dense(num_hidden_units, activation='relu')(y)
y = Dense(num_classes, activation='softmax')(y)
model = Model(x, y)

print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

# compile the model aganist the binary cross entropy loss
# and use SGD optimizer
model.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
print("[MESSAGE] Model is compiled.")

# Create Image Data Generator
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=10,
                             shear_range=0.1)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_x, augment=False)

print("[MESSAGE] Data Generator is created.")


# train the model
history = model.fit_generator(datagen.flow(train_x, train_Y, batch_size=64),
                              steps_per_epoch=len(train_x) / 64,
                              epochs=10,
                              validation_data=(test_x, test_Y))

print("[MESSAGE] Model is trained.")

# save the trained model
model.save("conv-net-fashion-mnist-trained.hdf5")

print("[MESSAGE] Model is saved.")

# visualize the ground truth and prediction
# take first 10 examples in the testing dataset
test_x_vis = test_x[:10]  # fetch first 10 samples
ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
# predict with the model
preds = np.argmax(model.predict(test_x_vis), axis=1).astype(np.int)


plt.figure()
for i in range(2):
  for j in range(5):
    plt.subplot(2, 5, i * 5 + j + 1)
    plt.imshow(np.squeeze(test_x[i * 5 + j]), cmap="gray")
    plt.title("Ground Truth: %s, \n Prediction %s" %
              (string.ascii_uppercase[ground_truths[i * 5 + j]],
               string.ascii_uppercase[preds[i * 5 + j]]))
plt.show()
