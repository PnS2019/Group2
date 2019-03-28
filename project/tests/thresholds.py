#!/usr/bin/python3
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread('../data/example2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

_, contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'OTSU', 'TRUNC', 'TOZERO', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [gray, thresh1, thresh3, thresh4, th2, th3]

min_size = 20
for i in range(6):
    _, contours, _ = cv2.findContours(images[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    colored = cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB)
    for contour in contours:
        #cv2.drawContours(colored, contour, -1, (255, 0, 0), -1)
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(colored, (x, y), (x + w, y + h), (0, 255, 0), 1)

    plt.subplot(2, 3, i + 1), plt.imshow(colored)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
