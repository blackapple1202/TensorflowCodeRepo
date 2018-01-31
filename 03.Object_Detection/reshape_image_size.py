import PIL
from PIL import Image
import pandas as pd
import shutil

# This python script reshape images

# TODO: IF YOU WANT TO CHANGE SETTINGS,
#       YOU MAY FIX THESE VARIABLES
SOURCE_IMAGE_PATH = 'images'
IMAGES_COUNT = 101
MAXIMUM_PIXEL = 500

for directory in [SOURCE_IMAGE_PATH]:
    for i in range(1, IMAGES_COUNT):
        # Open image on images directory
        im = Image.open('{}/{}.jpg'.format(directory, i))
        # Get Image Size
        s = im.size
        print('{}/{}.jpg is ({}, {})'.format(directory, i, s[0], s[1]))
        if s[0] <= MAXIMUM_PIXEL and s[1] <= MAXIMUM_PIXEL:
            continue

        # If width is bigger than height
        if s[0] >= s[1]:
            print('{}/{}.jpg is reshaping...'.format(directory, i))
            width_percent = (MAXIMUM_PIXEL / float(s[0]))
            height_size = int((float)(s[1]) * float(width_percent))
            im = im.resize((MAXIMUM_PIXEL, height_size), PIL.Image.ANTIALIAS)
        else:
            print('{}/{}.jpg is reshaping...'.format(directory, i))
            height_percent = (MAXIMUM_PIXEL / float(s[1]))
            width_size = int((float(s[0]) * float(height_percent)))
            im = im.resize((width_size, MAXIMUM_PIXEL), PIL.Image.ANTIALIAS)

        im.save('{}/{}.jpg'.format(directory, i))
        print('{}/{}.jpg is successfully reshape to ({}, {})'.format(directory, i, im.size[0], im.size[1]))

