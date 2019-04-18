#!/usr/bin/python3

from PIL import Image, ImageFont, ImageDraw
import string


def draw(text, size, color, background_color=(0, 0, 0)):
    fontPath = '/home/FreeSansBold.ttf'
    font = ImageFont.truetype(fontPath, size[0])
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


size = (200, 200)
background_color = (0, 0, 0)
letter_color = (255, 255, 255)

font = 'data/DejaVuSans-Bold.ttf'

for letter in string.ascii_letters:
    if letter in "ij":
        letter = letter.upper()
    image = draw(letter, size, letter_color)
    image.save("data/letters/{}.jpg".format(letter))
