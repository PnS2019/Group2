#!/usr/bin/python3

import cv2
import numpy as np

import matplotlib.pyplot as plt

# read the image file
img = cv2.imread("Lenna.png")  # put the lenna.png at the same directory as the script

# extract height and width of the image
height, width = img.shape[:2]
# resize the image
res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

# display the image
cv2.imshow('rescaled', res)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Transformations
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = img_grey.shape

# define translation matrix
# move 100 pixels on x-axis
# move 50 pixels on y-axis
M_trans = np.float32([[1, 0, 100], [0, 1, 50]])

# flip x-axis
M_flip = np.float32([[-1, 0, cols], [0, 1, 0]])

# rotate 30
M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)

# affine
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_aff = cv2.getAffineTransform(pts1, pts2)

# translate the image
dst_trans = cv2.warpAffine(img_grey, M_trans, (cols, rows))
dst_flip = cv2.warpAffine(img_grey, M_flip, (cols, rows))
dst_rot = cv2.warpAffine(img_grey, M_rot, (cols, rows))
dst_aff = cv2.warpAffine(img_grey, M_aff, (cols, rows))

# display the image
cv2.imshow('translation', dst_trans)
cv2.imshow('flip x-axis', dst_flip)
cv2.imshow('rotated 30 deg', dst_rot)
cv2.imshow('affine transformation', dst_aff)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Thresholds

# apply global thresholding
ret, th1 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)

# apply mean thresholding
# the function calculates the mean of a 11x11 neighborhood area for each pixel
# and subtract 2 from the mean
th2 = cv2.adaptiveThreshold(
    img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, 11, 2)

# apply Gaussian thresholding
# the function calculates a weights sum by using a 11x11 Gaussian window
# and subtract 2 from the weighted sum.
th3 = cv2.adaptiveThreshold(
    img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)

# display the processed images
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img_grey, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# Filtering

# prepare a 11x11 averaging filter
kernel = np.ones((11, 11), np.float32) / 121
dst = cv2.filter2D(img, -1, kernel)

# change image from BGR space to RGB space
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# display the result
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# perform laplacian filtering
laplacian = cv2.Laplacian(img_grey, cv2.CV_64F)
# find vertical edge
sobelx = cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, ksize=3)
# find horizontal edge
sobely = cv2.Sobel(img_grey, cv2.CV_64F, 0, 1, ksize=3)

plt.subplot(2, 2, 1), plt.imshow(img_grey, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

# Edges

# Find edge with Canny edge detection
edges = cv2.Canny(img, 100, 200)

# display results
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
