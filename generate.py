from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path

import numpy as np
#import cv2
import tensorflow as tf
import prettytensor as pt

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import model
import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', './tmp',
                           '''Directory where the model file is located.''')

GRID = (8, 8)

def init_variables(sess):
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    variables_to_restore = gen_vars
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception('No checkpoint file found')
    return

def main(argv=None):
    input.init_dataset_constants()
    num_images = GRID[0] * GRID[1]
    FLAGS.batch_size = num_images
    with tf.Graph().as_default():
        g_template = model.generator_template()
        z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.z_size])
        #np.random.seed(1337) # generate same random numbers each time
        noise = np.random.normal(size=(FLAGS.batch_size, FLAGS.z_size))
        with pt.defaults_scope(phase=pt.Phase.test):
            gen_images_op, _ = pt.construct_all(g_template, input=z)

        sess = tf.Session()
        init_variables(sess)
        gen_images, = sess.run([gen_images_op], feed_dict={z: noise})
        gen_images = (gen_images + 1) / 2

        sess.close()
        
        fig = plt.figure(1)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=GRID,
                         axes_pad=0.1)
        for i in xrange(num_images):
            im = gen_images[i]
            axis = grid[i]
            axis.axis('off')
            axis.imshow(im)

        plt.show()
        fig.savefig('montage.png', dpi=100, bbox_inches='tight')

if __name__ == '__main__':
    tf.app.run()

