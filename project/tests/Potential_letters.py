import os
import cv2 as cv
import numpy as np
import copy
import string

potential_letters = []

upper_bound_rectangle_area = 18000
lower_bound_rectangle_area = 500

# defining a sort function


def xCoo(arr):
    return arr[0][0]


def yCoo(arr):
    return arr[0][1]


def letters(picture):
    potential_letters = []
    # color detection
    lower_blue = np.array([100, 40, 0])
    upper_blue = np.array([135, 255, 255])

    # converting colorspace:
    picture = cv.resize(picture, (1920, 1080))
    hsv = cv.cvtColor(picture, cv.COLOR_BGR2HSV)
    # filtered version of Picture:
    #hsv = cv.GaussianBlur(hsv,(3,3),2)

    filtered = cv.inRange(hsv, lower_blue, upper_blue)
    filtered = cv.bitwise_not(filtered)

    # contour detection:

    _, ctrs, hier = cv.findContours(filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    picture = cv.cvtColor(filtered, cv.COLOR_GRAY2RGB)

    # creting an array of rectangles with areas
    rects_and_area = [[cv.boundingRect(ctr), cv.contourArea(ctr)] for ctr in ctrs]

    # sorting the areas and rectangles from left to right
    rects_and_area.sort(key=xCoo)

    for rectangle in rects_and_area:

        if(rectangle[1] < upper_bound_rectangle_area and rectangle[1] > lower_bound_rectangle_area):

            out_cropped = picture[rectangle[0][1]:rectangle[0][1] + rectangle[0][3],
                                  rectangle[0][0]:rectangle[0][0] + rectangle[0][2]]
            out_cropped = cv.resize(out_cropped, (32, 32))

            potential_letters.append(out_cropped)
    return potential_letters
