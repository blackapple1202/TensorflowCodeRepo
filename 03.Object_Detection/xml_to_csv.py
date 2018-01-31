

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# TODO: IF YOU WANT TO CHANGE [NAME or PATH] of [SOURCE or OUTPUT] files,
#       YOU MAY FIX THESE VARIABLES
SOURCE_IMAGE_PATH = 'images/{}'
OUTPUT_CSV_PATH = 'data/{}_labels.csv'
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in [TRAIN_FOLDER, TEST_FOLDER]:
	    image_path = os.path.join(os.getcwd(), SOURCE_IMAGE_PATH.format(directory))
	    xml_df = xml_to_csv(image_path)
	    xml_df.to_csv(OUTPUT_CSV_PATH.format(directory), index=None)
	    print('Successfully converted all xml files on {} to '.format(SOURCE_IMAGE_PATH.format(directory)) + OUTPUT_CSV_PATH.format(directory))


main()
