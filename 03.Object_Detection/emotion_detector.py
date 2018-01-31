
# coding: utf-8


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

#                                                                                                   #
# TODO: FIX THIS PART OF CODE, IF YOU WANT TO SET YOUR FONT CONFIGURATION                           #
#                                                                                                   #
VIDEO_SPEED = 1
FONT_COLOR = (255, 0 , 0)
FONT_THICKNESS = 2
FONT_SIZE = 1

#                                                                                                   #
# TODO: FIX THIS PART OF CODE, IF YOU WANT TO SET YOUR CV WINDOW RESOLUTION                         #
#                                                                                                   #
CV_WINDOW_WIDTH = 1600
CV_WINDOW_HEIGHT = 900


#                                                                                                   #
# TODO: FIX THIS PART OF CODE, IF YOU WANT TO SET YOUR CAMERA                                       #
#                                                                                                   #
cap = cv2.VideoCapture('test01.avi')
#cap = cv2.VideoCapture(0)

#                                                                                                   #
# TODO: FIX THIS PART OF CODE, IF YOU CHANGE THE MODEL AND LABEL                                    #
#                                                                                                   #
MODEL_NAME = 'face_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'face-detection.pbtxt')
NUM_CLASSES = 7

#                                                                                                   #
# TODO: FIX THIS PART OF CODE, IF YOU CHANGE THE LABEL                                              #
#                                                                                                   #
NAME_CLASSES = ['N/A', 'Neutral', 'Sad', 'Surprised', 'Happy', 'Fear', 'Angry', 'Disgust']



if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


from utils import label_map_util

from utils import visualization_utils as vis_util



# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection for images, not the streaming
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)




with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    # Use OpenCV python for showing a detection
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)

      #                                                                                                   #
      # TODO: FIX THIS PART OF CODE, IF YOU CHANGE THE LABEL                                              #
      #                                                                                                   #
      # Extracting score from face
      SCORE_CLASSES = {'N/A': 0, 'Neutral': 0, 'Sad': 0, 'Surprised': 0, 'Happy': 0, 'Fear': 0, 'Angry':0, 'Disgust':0}
      max_boxes_to_draw = 20
      if not max_boxes_to_draw:
          max_boxes_to_draw = np.squeeze(boxes).shape[0]
      for i in range(min(max_boxes_to_draw, np.squeeze(boxes).shape[0])):
          if np.squeeze(scores) is None or np.squeeze(scores)[i] > .5:
              if np.squeeze(classes).astype(np.int32)[i] in category_index.keys():
                  class_name = category_index[np.squeeze(classes).astype(np.int32)[i]]['name']
              else:
                  class_name = 'N/A'
              if scores is not None:
                  SCORE_CLASSES[class_name] = (int(100 * np.squeeze(scores)[i]))

      for i in range (1, NUM_CLASSES + 1):
        cv2.putText(image_np, '{}'.format(NAME_CLASSES[i]), (10, 15 * i + 5), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
        cv2.rectangle(image_np, (100, 15 * i), ((100 + SCORE_CLASSES[NAME_CLASSES[i]]), 15 * i + 10), FONT_COLOR, -1)
        cv2.putText(image_np, '{}'.format(SCORE_CLASSES[NAME_CLASSES[i]]), (210, 15 * i + 5), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)

      cv2.imshow('ita-tech', cv2.resize(image_np, (CV_WINDOW_WIDTH , CV_WINDOW_HEIGHT)))
      if cv2.waitKey(VIDEO_SPEED) & 0xFF == ord('q'):
          cap.release()
          cv2.destroyAllWindows()
          break


