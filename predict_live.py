import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from pathlib import Path
import cv2 as cv
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ < '1.4.0':
	raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

TF_OBDET_ROOT = Path('../models/')

sys.path.append(str(TF_OBDET_ROOT/'research/'))
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# utils
def load_image_into_numpy_array(image):
	if image.mode == 'L':
		image = image.convert('RGB')
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)


# # What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# MODEL_DIR = Path('models/')
# MODEL_TAR = MODEL_DIR/MODEL_FILE

# TF_OBDET_DIR = Path('../tf_models/research/object_detection')


# if not MODEL_TAR.exists():
# 	opener = urllib.request.URLopener()
# 	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, str(MODEL_TAR))

# tar_file = tarfile.open(str(MODEL_TAR))
# for file in tar_file.getmembers():
# 	file_name = os.path.basename(file.name)
# 	tar_file.extract(file, MODEL_DIR)
	


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'models/mobilenet_pens_ig/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/mobilenet_pens_ig/labelmap.pbtxt' #os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(str(PATH_TO_CKPT), 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(str(PATH_TO_LABELS))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
cap = cv.VideoCapture(0)
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
                for i in range(800):
                        ret, image_np = cap.read()
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
			# Actual detection.
                        (boxes, scores, classes, num) = sess.run(
			  [detection_boxes, detection_scores, detection_classes, num_detections],
			  feed_dict={image_tensor: image_np_expanded})
                        print(classes)
                        print(category_index)
			# Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
			  image_np,
			  np.squeeze(boxes),
			  np.squeeze(classes).astype(np.int32),
			  np.squeeze(scores),
                          category_index={1:{'name':'Pen'}},
                          min_score_thresh=0.8,
			  use_normalized_coordinates=True,
			  line_thickness=8)
                        
                        cv.imshow('object detection', cv.resize(image_np, (800,600)))
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            cv.destroyAllWindows()
                            break
