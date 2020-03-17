import tensorflow as tf


from PIL import Image
import sys
sys.path.append('../models/research')
from object_detection.utils import dataset_util

dir1= "/home/piyumal/Desktop/tfrecord_2/train.tfrecord"
dir2= "/home/piyumal/Desktop/tfrecord_1/train.record"


# Create dataset from multiple .tfrecord files
list_of_tfrecord_files = [dir1, dir2]
dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

# Save dataset to .tfrecord file
filename = 'merged.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(dataset)
