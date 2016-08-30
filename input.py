from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import time

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './datasets/anime_faces',
                           'Training data directory.')
tf.app.flags.DEFINE_string('dataset', 'custom',
                           'One of: custom, cifar')

def read_and_decode_cifar(filename_queue):
    label_bytes = 1
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [depth, height, width])
    image = tf.transpose(depth_major, [1, 2, 0])
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return image

def read_and_decode1(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'file_bytes': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([NUM_TAGS], tf.float32),
        })

    # decode the jpeg image
    image = tf.image.decode_jpeg(features['file_bytes'], channels=3, try_recover_truncated=True)

    # Convert to float image
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    # convert to grayscale if needed
    if CHANNELS == 1:
        image = tf.reduce_mean(image, reduction_indices=[2], keep_dims=True)

    image.set_shape([None, None, None])

    shape = tf.cast(tf.shape(image), tf.float32)
    height_pad = tf.maximum(tf.ceil((96 - shape[0]) / 2), 0)
    height_pad = tf.reshape(height_pad, [1,1])
    width_pad = tf.maximum(tf.ceil((96 - shape[1]) / 2), 0)
    width_pad = tf.reshape(width_pad, [1,1])
    height_pad = tf.tile(height_pad, [1, 2])
    width_pad = tf.tile(width_pad, [1, 2])
    paddings = tf.concat(0, [height_pad, width_pad, tf.zeros([1, 2])])
    paddings = tf.cast(paddings, tf.int32)
    image = tf.pad(image, paddings)

    # randomly crop out a section
    image = tf.random_crop(image, [96, 96, CHANNELS])

    # downsample
    image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE, method=tf.image.ResizeMethod.AREA)
    #image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE, method=tf.image.ResizeMethod.BICUBIC)

    # randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)

    label = features['label']
    label = tf.slice(label, [0], [NUM_TAGS_TO_USE])
    
    return image, label

def read_and_decode2(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'file_bytes': tf.FixedLenFeature([], tf.string),
        })

    # decode the png image
    image = tf.image.decode_png(features['file_bytes'], channels=3)

    # Convert to float image
    image = tf.cast(image, tf.float32)

    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    # convert to grayscale if needed
    if CHANNELS == 1:
        image = tf.reduce_mean(image, reduction_indices=[2], keep_dims=True)

    # normalize
    image = image * (2. / 255) - 1

    return image

def inputs():
    if FLAGS.dataset == 'cifar':
        filenames = [os.path.join(FLAGS.data_dir, 'cifar', 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        filename_queue = tf.train.string_input_producer(filenames)
        image = read_and_decode_cifar(filename_queue)
    elif FLAGS.dataset == 'custom':
        filenames = tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        filename_queue = tf.train.string_input_producer(filenames)
        image = read_and_decode2(filename_queue)
    else:
        raise NotImplemented()

    # randomly flip
    image = tf.image.random_flip_left_right(image)

    num_preprocess_threads = 4

    # ensure that the random shuffling has good mixing properties
    min_queue_examples = 500
    format_str = ('Filling queue with {} images before training. '
                  'This might take a while.')
    print(format_str.format(min_queue_examples))

    images = tf.train.shuffle_batch(
        [image],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=3*min_queue_examples,
        min_after_dequeue=min_queue_examples)

    # display training images in visualizer
    tf.image_summary('images', images, max_images=FLAGS.batch_size, name='images_summary')

    return images

def init_dataset_constants():
    global IMAGE_SIZE
    global NUM_LEVELS
    global CHANNELS
    if FLAGS.dataset == 'cifar':
        IMAGE_SIZE = 32
        NUM_LEVELS = 3
        CHANNELS = 3
    elif FLAGS.dataset == 'custom':
        IMAGE_SIZE = 64
        NUM_LEVELS = 4
        CHANNELS = 3
    else:
        raise NotImplemented()

