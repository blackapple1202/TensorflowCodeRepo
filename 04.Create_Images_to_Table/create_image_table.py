import PIL
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import shutil
import os.path
from pathlib import Path






# Table Configure Variables
IMAGE_START_NUMBER = 1
IMAGE_END_NUMBER = 121
TABLE_IM_PIXEL = 500
TABLE_IM_WIDTH_NUMBER = 10
TABLE_IM_HEIGHT_NUMBER = 10
INPUT_FOLDER = 'images'
INPUT_IMAGE_FILE_NAME = '{}_crop.png'
OUTPUT_FOLDER = 'data'
OUTPUT_IMAGE_FILE_NAME = '{}_table.jpg'

# Table Boundary Configure
BOUNDARY_PADDING_PIXEL = {'top': 4, 'bottom': 4, 'left': 4, 'right': 4}

# CSV Configure
LABEL = 'face'
OUTPUT_CSV_FILE_NAME = '{}_labels.csv'

# Training(True) or Testing(False)?
DATA_USAGE = True








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

def draw_boundary_on_table_image(tableImg, source_image_pixel, boundary_padding):
    DrawImg = ImageDraw.Draw(tableImg)
    for i in range(0, TABLE_IM_WIDTH_NUMBER):
        for j in range(0, TABLE_IM_HEIGHT_NUMBER):
            xmin = j * source_image_pixel + boundary_padding['left']
            ymin =  i * source_image_pixel + boundary_padding['top']
            xmax = j * source_image_pixel + source_image_pixel - boundary_padding['right']
            ymax = i * source_image_pixel + source_image_pixel - boundary_padding['bottom']
            DrawImg.rectangle([(xmin, ymin), (xmax, ymax)], fill = None, outline='green')
    del DrawImg

def save_boundaries_to_csv(path, input_image_list):
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    images_df = pd.DataFrame(input_image_list, columns=column_name)
    images_df.to_csv(path, index=None)

def append_boundary_to_csv(output_image_list, filename, width, height, label, xmin, ymin, xmax, ymax):
    value = (filename, width, height, label, xmin, ymin, xmax, ymax)
    output_image_list.append(value)

def main():

    if DATA_USAGE:
        usage = 'train'
    else:
        usage = 'test'

    image_list = []

    SOURCE_IM_PIXEL = (TABLE_IM_PIXEL / TABLE_IM_WIDTH_NUMBER)
    tableImage = Image.new('RGB', (TABLE_IM_PIXEL,TABLE_IM_PIXEL))
    IMAGES_COUNT = IMAGE_START_NUMBER

    for directory in [INPUT_FOLDER]:
        for i in range(0, TABLE_IM_WIDTH_NUMBER):
            for j in range(0, TABLE_IM_HEIGHT_NUMBER):
                # Open image on images directory
                IMAGES_COUNT = IMAGES_COUNT + 1
                while not check_image_with_pil('{}/{}'.format(directory, INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT))):
                    IMAGES_COUNT = IMAGES_COUNT + 1
                    if IMAGES_COUNT > IMAGE_END_NUMBER:
                        save_table_image('{}/{}'.format(OUTPUT_FOLDER, OUTPUT_IMAGE_FILE_NAME.format(usage)), tableImage)
                        print('Successfully save images to table')
                        csv_path = '{}/{}'.format(OUTPUT_FOLDER, OUTPUT_CSV_FILE_NAME.format(usage))
                        save_boundaries_to_csv(csv_path, image_list)
                        print('Successfully save boundaries to csv')

                        draw_boundary_on_table_image(tableImage, SOURCE_IM_PIXEL,BOUNDARY_PADDING_PIXEL)
                        show_table_image(tableImage)
                        print('End of file is {}'.format(INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT)))
                        return


                im = Image.open('{}/{}'.format(directory,  INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT)))
                im = ImageOps.expand(im, border=(int)(SOURCE_IM_PIXEL*0.01), fill='white')
                im.thumbnail((SOURCE_IM_PIXEL, SOURCE_IM_PIXEL))

                append_boundary_to_csv(image_list,
                                       OUTPUT_IMAGE_FILE_NAME.format(usage),
                                       TABLE_IM_PIXEL,
                                       TABLE_IM_PIXEL,
                                       LABEL,
                                       (j * SOURCE_IM_PIXEL) + BOUNDARY_PADDING_PIXEL['left'],
                                       (i * SOURCE_IM_PIXEL) + BOUNDARY_PADDING_PIXEL['top'],
                                       (j * SOURCE_IM_PIXEL) + SOURCE_IM_PIXEL - BOUNDARY_PADDING_PIXEL['right'],
                                       (i * SOURCE_IM_PIXEL) + SOURCE_IM_PIXEL - BOUNDARY_PADDING_PIXEL['bottom'])

                tableImage.paste(im, ((j * SOURCE_IM_PIXEL),(i * SOURCE_IM_PIXEL)))

    # Save process
    save_table_image('{}/{}'.format(OUTPUT_FOLDER, OUTPUT_IMAGE_FILE_NAME.format(usage)), tableImage)
    print('Successfully save images to table')
    csv_path = '{}/{}'.format(OUTPUT_FOLDER, OUTPUT_CSV_FILE_NAME.format(usage))
    save_boundaries_to_csv(csv_path, image_list)
    print('Successfully save boundaries to csv')

    # Show process
    draw_boundary_on_table_image(tableImage, SOURCE_IM_PIXEL,BOUNDARY_PADDING_PIXEL)
    show_table_image(tableImage)

    print('End of file is {}'.format(INPUT_IMAGE_FILE_NAME.format(IMAGES_COUNT)))


main()