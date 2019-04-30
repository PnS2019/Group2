#!/usr/bin/python3

# Standardize images across the dataset, mean=0, stdev=1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras import backend as K
import numpy as np
import string
import os
import cv2

# load data
X = []
y = []

letters = string.ascii_uppercase

for image in os.listdir("data/letters/"):
    if image.endswith(".jpg"):
        im = cv2.imread("data/letters/" + image)
        letter = image[0].upper()

        index = letters.index(letter)

        X.append(im)
        y.append(index)

X = np.array(X)
y = np.array(y)

perm = np.random.permutation(len(X))

X = X[perm]
y = y[perm]


split = .2

test_amount = int(len(X) * split)
X_train = np.array(X[test_amount:])
y_train = np.array(y[test_amount:])


X_test = np.array(X[:test_amount])
y_test = np.array(y[:test_amount])

print(X_train.shape)
# define data preparation
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=10,
                             shear_range=0.1,
                             validation_split=0.2)

# fit parameters from data
datagen.fit(X_train)

letter_counts = [0] * 26

cnt = 0

# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=10):
    # create a grid of 3x3 images
    for i in range(0, 10):
        name = "data/letters/augmented/{}_{}.jpg".format(string.ascii_uppercase[y_batch[i]],
                                                         letter_counts[y_batch[i]])
        letter_counts[y_batch[i]] += 1
        cv2.imwrite(name, X_batch[i])
    cnt += 10
    print("\r{} Images created.".format(cnt), end=" " * 5)
