#!/usr/bin/python3
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread('../data/example2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

rectangles = []
for contour in contours:
    #cv2.drawContours(img, contour, -1, (255, 0, 0), -1)
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    rectangles.append((x, y, w, h))

thresh_horiz = 10
thresh_vert = 10

clusters = []
for i in range(len(rectangles)):
    for j in range(len(rectangles)):
        if i != j:
            left = rectangles[i]
            right = rectangles[i]
            if abs(left[0] + left[2] - right[0]) <= thresh_horiz and \
                    abs(left[1] - abs(right[1]) <= thresh_vert):
                clusters.append(left)

print(len(clusters))

cv2.imshow("binary", img)
cv2.waitKey(0)
