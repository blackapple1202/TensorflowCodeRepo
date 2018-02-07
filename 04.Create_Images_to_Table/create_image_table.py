import PIL
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import shutil
import os.path
import random
from pathlib import Path




############### CONFIGURE ########################

# Table Configure Variables

# Image Size Configuration
IMAGE_START_NUMBER = 1
IMAGE_END_NUMBER = 200
TABLE_IM_PIXEL = 480
TABLE_IM_WIDTH_NUMBER = 4
TABLE_IM_HEIGHT_NUMBER = 4


# Image Background Configuration
BACKGROUND_START_NUMBER = 1
BACKGROUND_END_NUMBER = 16
BACKGROUND_FOLDER = 'backgrounds'
BACKGROUND_IMAGE_FILE_NAME = '{}_background.jpg'

# Set input path and output path
INPUT_FOLDER = 'images'
INPUT_IMAGE_FILE_NAME = '{}_crop.png'

OUTPUT_FOLDER = 'data'
OUTPUT_IMAGE_FILE_NAME = '{}_table{}.jpg'
OUTPUT_CLONE_FOLDER = 'data/clone'

# Set REPETITION number of extraction
EXTRACT_OUTPUT_INDEX_MIN = 181
EXTRACT_OUTPUT_INDEX_MAX = 240
# REPETITION NUMBER = EXTRACT_OUTPUT_INDEX_MAX - EXTRACT_OUTPUT_INDEX_MIN + 1

# Toggle options
TOGGLE_BACKGROUND = True
TOGGLE_SHUFFLE_BACKGROUND = False
TOGGLE_SHUFFLE_IMAGE = True
TOGGLE_CSV_TO_SAVE_INDIVIDUAL = False
TOGGLE_CLONE_IMAGE_TO_SHOW = False
TOGGLE_CLONE_IMAGE_TO_SAVE = True
OUTPUT_CLONE_IMAGE_FILE_NAME = 'include_boundaries_{}_table{}.jpg'

# Set index of EXTRACT_MODE to OUTPUT_IMAGE_EXTRACT_MODE
# Default is same as 'all'
EXTRACT_MODE = ['default', 'all', 'odd', 'even' , 'random']
RANDOM_START_RANGE_MIN = 0
RANDOM_START_RANGE_MAX = 3
RANDOM_INCREMENT_RANGE_MIN = 2
RANDOM_INCREMENT_RANGE_MAX = 6
OUTPUT_IMAGE_EXTRACT_MODE = EXTRACT_MODE[4]

# Table Boundary Configure
BOUNDARY_PADDING_PIXEL = {'top': 4, 'bottom': 4, 'left': 4, 'right': 4}

# CSV Configure
LABEL = 'face'
OUTPUT_CSV_FILE_NAME = '{}_labels{}.csv'

# Extract Training(True) or Testing(False)?
DATA_USAGE = True


###################################################


start_step = 0
increment_step = 1

def check_image_with_pil(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True

def show_table_image(tableImg):
    tableImg.show()

def save_table_image(path , tableImg):
    tableImg.save(path)

def save_boundaries_to_csv(path, input_image_list):
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    images_df = pd.DataFrame(input_image_list, columns=column_name)
    images_df.to_csv(path, index=None)

def append_boundary_to_csv(output_image_list, filename, width, height, label, xmin, ymin, xmax, ymax):
    value = (filename, width, height, label, xmin, ymin, xmax, ymax)
    output_image_list.append(value)

def extract(repeat_index, background_index, all_image_list):

    if DATA_USAGE:
        usage = 'train'
    else:
        usage = 'test'

    image_list = []

    SOURCE_IM_PIXEL = (TABLE_IM_PIXEL / TABLE_IM_WIDTH_NUMBER)
    tableImage = Image.new('RGB', (TABLE_IM_PIXEL,TABLE_IM_PIXEL))
    IMAGES_COUNT = IMAGE_START_NUMBER

    clone_tableImage = Image.new('RGB', (TABLE_IM_PIXEL,TABLE_IM_PIXEL))
    DrawImg = ImageDraw.Draw(clone_tableImage)

    if TOGGLE_BACKGROUND:
        background = Image.open('{}/{}'.format(BACKGROUND_FOLDER, BACKGROUND_IMAGE_FILE_NAME.format(background_index)))
        background = background.resize((TABLE_IM_PIXEL, TABLE_IM_PIXEL), PIL.Image.ANTIALIAS)
        tableImage.paste(background, (0, 0))
        clone_tableImage.paste(background, (0, 0))

    if not RANDOM_INCREMENT_RANGE_MIN > 0 or not RANDOM_INCREMENT_RANGE_MAX > RANDOM_INCREMENT_RANGE_MIN:
        print('RANDOM_INCREMENT_RANGE should be set properly')
        return

    for directory in [INPUT_FOLDER]:
        for i in range(0, TABLE_IM_WIDTH_NUMBER):
            start_step = 0
            increment_step = 1
            if OUTPUT_IMAGE_EXTRACT_MODE == 'all':
                start_step = 0
                increment_step = 1
            elif OUTPUT_IMAGE_EXTRACT_MODE == 'odd':
                if i % 2 == 0:
                    start_step = 1
                else:
                    start_step = 0
                increment_step = 2
            elif OUTPUT_IMAGE_EXTRACT_MODE == 'even':
                if i % 2 == 0:
                    start_step = 0
                else:
                    start_step = 1
                increment_step = 2
            elif OUTPUT_IMAGE_EXTRACT_MODE == 'random':
                start_step = random.randrange(RANDOM_START_RANGE_MIN, RANDOM_START_RANGE_MAX)
                increment_step = random.randrange(RANDOM_INCREMENT_RANGE_MIN, RANDOM_INCREMENT_RANGE_MAX)
            for j in range(start_step, TABLE_IM_HEIGHT_NUMBER, increment_step):

                # Open image on images directory
                if TOGGLE_SHUFFLE_IMAGE:
                    IMAGES_COUNT = random.randrange(IMAGE_START_NUMBER, IMAGE_END_NUMBER)
                else:
                    IMAGES_COUNT = IMAGES_COUNT + 1

                # If image is not exist on folder
                while not check_image_with_pil('{}/{}'.format(directory, INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT))):
                    # Skip to next index
                    if TOGGLE_SHUFFLE_IMAGE:
                        IMAGES_COUNT = random.randrange(IMAGE_START_NUMBER, IMAGE_END_NUMBER)
                    else:
                        IMAGES_COUNT = IMAGES_COUNT + 1
                    # If image index is overwhelmed the end number
                    if IMAGES_COUNT > IMAGE_END_NUMBER:
                        # Save process35f
                        save_table_image('{}/{}'.format(OUTPUT_FOLDER, OUTPUT_IMAGE_FILE_NAME.format(usage,repeat_index)), tableImage)
                        print('Successfully save images to table')
                        if TOGGLE_CSV_TO_SAVE_INDIVIDUAL:
                            csv_path = '{}/{}'.format(OUTPUT_FOLDER, OUTPUT_CSV_FILE_NAME.format(usage,repeat_index))
                            save_boundaries_to_csv(csv_path, image_list)
                            print('Successfully save boundaries to csv')
                        if TOGGLE_CLONE_IMAGE_TO_SAVE:
                            save_table_image('{}/{}'.format(OUTPUT_CLONE_FOLDER, OUTPUT_CLONE_IMAGE_FILE_NAME.format(usage,repeat_index)), clone_tableImage)

                        # Show process
                        if TOGGLE_CLONE_IMAGE_TO_SHOW:
                            show_table_image(clone_tableImage)
                        print('End of file is {}'.format(INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT)))
                        # End of script
                        return


                im = Image.open('{}/{}'.format(directory,  INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT)))
                im = ImageOps.expand(im, border=(int)(SOURCE_IM_PIXEL*0.01), fill='white')
                im = im.resize((TABLE_IM_PIXEL, TABLE_IM_PIXEL), PIL.Image.ANTIALIAS)

                im.thumbnail((SOURCE_IM_PIXEL, SOURCE_IM_PIXEL))

                xmin = (j * SOURCE_IM_PIXEL) + BOUNDARY_PADDING_PIXEL['left']
                ymin = (i * SOURCE_IM_PIXEL) + BOUNDARY_PADDING_PIXEL['top']
                xmax = (j * SOURCE_IM_PIXEL) + SOURCE_IM_PIXEL - BOUNDARY_PADDING_PIXEL['right']
                ymax = (i * SOURCE_IM_PIXEL) + SOURCE_IM_PIXEL - BOUNDARY_PADDING_PIXEL['bottom']

                append_boundary_to_csv(image_list,
                                       OUTPUT_IMAGE_FILE_NAME.format(usage, repeat_index),
                                       TABLE_IM_PIXEL,
                                       TABLE_IM_PIXEL,
                                       LABEL,
                                       xmin,
                                       ymin,
                                       xmax,
                                       ymax)

                append_boundary_to_csv(all_image_list,
                                       OUTPUT_IMAGE_FILE_NAME.format(usage, repeat_index),
                                       TABLE_IM_PIXEL,
                                       TABLE_IM_PIXEL,
                                       LABEL,
                                       xmin,
                                       ymin,
                                       xmax,
                                       ymax)

                tableImage.paste(im, ((j * SOURCE_IM_PIXEL),(i * SOURCE_IM_PIXEL)))
                clone_tableImage.paste(im, ((j * SOURCE_IM_PIXEL),(i * SOURCE_IM_PIXEL)))
                DrawImg.rectangle([(xmin, ymin), (xmax, ymax)], fill=None, outline='green')

    # Save process
    save_table_image('{}/{}'.format(OUTPUT_FOLDER, OUTPUT_IMAGE_FILE_NAME.format(usage,repeat_index)), tableImage)
    print('Successfully save images to table')

    if TOGGLE_CSV_TO_SAVE_INDIVIDUAL:
        csv_path = '{}/{}'.format(OUTPUT_FOLDER, OUTPUT_CSV_FILE_NAME.format(usage,repeat_index))
        save_boundaries_to_csv(csv_path, image_list)
        print('Successfully save boundaries to csv')
    if TOGGLE_CLONE_IMAGE_TO_SAVE:
        save_table_image('{}/{}'.format(OUTPUT_CLONE_FOLDER, OUTPUT_CLONE_IMAGE_FILE_NAME.format(usage,repeat_index)), clone_tableImage)

    # Show process
    if TOGGLE_CLONE_IMAGE_TO_SHOW:
        show_table_image(clone_tableImage)
    print('End of file is {}'.format(INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT)))
    # End of Script

def main():
    if not EXTRACT_OUTPUT_INDEX_MIN > 0 or not EXTRACT_OUTPUT_INDEX_MAX >= EXTRACT_OUTPUT_INDEX_MIN:
        print('EXTRACT_OUTPUT_INDEX should be set properly')
        return
    background_index = 0
    image_list = []
    for i in range(EXTRACT_OUTPUT_INDEX_MIN, EXTRACT_OUTPUT_INDEX_MAX + 1):
        if TOGGLE_SHUFFLE_BACKGROUND:
            background_index = random.randrange(BACKGROUND_START_NUMBER, BACKGROUND_END_NUMBER)
        else:
            background_index = background_index + 1;
            if(background_index >= BACKGROUND_END_NUMBER):
                background_index = BACKGROUND_START_NUMBER
        extract(i, background_index, image_list)

    if DATA_USAGE:
        usage = 'train'
    else:
        usage = 'test'
    csv_path = '{}/{}'.format(OUTPUT_FOLDER, OUTPUT_CSV_FILE_NAME.format(usage, ''))
    save_boundaries_to_csv(csv_path, image_list)

main()