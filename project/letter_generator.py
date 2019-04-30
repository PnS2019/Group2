#!/usr/bin/python3

from PIL import Image, ImageFont, ImageDraw
import string
import os
from skimage.util import random_noise
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def draw(text, size, color, font, background_color=(0, 0, 0)):
    font = ImageFont.truetype(font, size[0])
    size2 = font.getsize(text)
    im = Image.new('RGB', size2, background_color)
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), text, font=font, fill=color)

    pixels = im.load()
    width, height = im.size
    max_x = max_y = 0
    min_y = height
    min_x = width

    # find the corners that bound the letter by looking for
    # non-transparent pixels
    for x in range(width):
        for y in range(height):
            p = pixels[x, y]
            if p != background_color:
                min_x = min(x, min_x)
                min_y = min(y, min_y)
                max_x = max(x, max_x)
                max_y = max(y, max_y)
    cropped = im.crop((min_x, min_y, max_x, max_y))
    return cropped.resize(size)


datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=10,
                             shear_range=0.1,
                             validation_split=0.2)

size = (32, 32)
background_color = (0, 0, 0)
letter_color = (255, 255, 255)

fonts = []
for font in os.listdir("data/fonts"):
    fonts.append("data/fonts/" + font)

for letter in string.ascii_letters:
    print("\r" + letter, end="", flush=True)
    if letter in "ij":
        letter = letter.upper()
    for i in range(len(fonts)):

        image = draw(letter, size, letter_color, fonts[i])
        image.save("data/letters/{}_{}.jpg".format(letter, i))
