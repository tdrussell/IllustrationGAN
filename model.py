from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
import prettytensor as pt
import numpy as np

import custom_ops
from custom_ops import leaky_rectify

import input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'default',
                           'Only default supported currently.')
tf.app.flags.DEFINE_integer('gen_fc_layers', 4,
                            'Number of fully connected layers in the generator.')
tf.app.flags.DEFINE_integer('gen_fc_size', 1024,
                            'Number of units to use in generator fully connected layers.')
tf.app.flags.DEFINE_integer('discrim_fc_layers', 3,
                            'Number of fully connected layers in the discriminator.')
tf.app.flags.DEFINE_integer('discrim_fc_size', 1024,
                            'Number of units to use in discriminator fully connected layers.')
tf.app.flags.DEFINE_integer('gen_filter_base', 64,
                            'Number of filters to use in lowest generator conv layer.')
tf.app.flags.DEFINE_integer('discrim_filter_base', 64,
                            'Number of filters to use in lowest discriminator conv layer.')
tf.app.flags.DEFINE_integer('z_size', 100,
                            'Size of the input distribution to the generator.')
tf.app.flags.DEFINE_float('keep_prob', 0.5,
                          'Probability of keeping values in dropout layers.')

# configuration options
optimizer = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.5)
gen_optimizer = lambda: optimizer(FLAGS.learning_rate)
discrim_optimizer = lambda: optimizer(FLAGS.learning_rate)
gen_activation_fn = tf.nn.relu
discrim_activation_fn = leaky_rectify

def _activation_summary(x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

'''
def reshape_for_sequence(tensor):
    shape = tensor.get_shape().as_list()
    height = shape[1]
    width = shape[2]
    depth = shape[3]
    result = (
        pt.wrap(tensor)
        .apply(tf.transpose, [0, 3, 2, 1])
        .reshape([FLAGS.batch_size, TIME_STEPS, -1])
        .apply(tf.transpose, [1, 0, 2])
        .reshape([-1, height*width*depth/TIME_STEPS])
    )
    return result

def reshape_from_sequence(tensor, side, depth):
    result = (
        pt.wrap(tensor)
        .reshape([TIME_STEPS, FLAGS.batch_size, -1])
        .apply(tf.transpose, [1, 0, 2])
        .reshape([FLAGS.batch_size, depth, side, side])
        .apply(tf.transpose, [0, 3, 2, 1])
    )
    return result
'''

def generator_template():
    starting_size = int(input.IMAGE_SIZE / (2 ** input.NUM_LEVELS))
    num_filters = FLAGS.gen_filter_base * (2 ** input.NUM_LEVELS)
    with tf.variable_scope('generator'):
        tmp = pt.template('input')
        for i in xrange(FLAGS.gen_fc_layers - 1):
            tmp = tmp.fully_connected(FLAGS.gen_fc_size).apply(gen_activation_fn)
        tmp = tmp.fully_connected(starting_size*starting_size*num_filters/2).apply(gen_activation_fn)
        features = tmp

        tmp = tmp.reshape([FLAGS.batch_size, starting_size, starting_size, num_filters/2])
        for i in xrange(input.NUM_LEVELS):
            num_filters = int(num_filters / 2)
            tmp = (
                tmp
                .upsample_conv(5, num_filters)
                #.custom_deconv2d(num_filters)
                .batch_normalize()
                .apply(gen_activation_fn)
            )
        tmp = tmp.conv2d(5, input.CHANNELS).apply(tf.nn.tanh)
        output = tmp

        z_prediction = (
            features
            .fully_connected(FLAGS.gen_fc_size)
            .apply(gen_activation_fn)
            .fully_connected(FLAGS.gen_fc_size)
            .apply(gen_activation_fn)
            .fully_connected(FLAGS.gen_fc_size)
            .apply(gen_activation_fn)
            .fully_connected(FLAGS.z_size)
        )

    return output, z_prediction

def discriminator_template():
    num_filters = FLAGS.discrim_filter_base
    with tf.variable_scope('discriminator'):
        tmp = pt.template('input')
        for i in xrange(input.NUM_LEVELS):
            if i > 0:
                tmp = tmp.dropout(FLAGS.keep_prob)
            tmp = tmp.conv2d(5, num_filters)
            if i > 0:
                tmp = tmp.batch_normalize()
            tmp = tmp.apply(discrim_activation_fn).max_pool(2, 2)
            num_filters *= 2
        tmp = tmp.flatten()
        features = tmp

        minibatch_discrim = features.minibatch_discrimination(100)

        for i in xrange(FLAGS.discrim_fc_layers-1):
            tmp = tmp.fully_connected(FLAGS.discrim_fc_size).apply(discrim_activation_fn)
        tmp = tmp.concat(1, [minibatch_discrim]).fully_connected(1)
        output = tmp

    return output
        
def losses(real_images):
    # get z
    z = tf.truncated_normal([FLAGS.batch_size, FLAGS.z_size], stddev=1) 
    #z = tf.random_uniform([FLAGS.batch_size, FLAGS.z_size], minval=-1, maxval=1)

    d_template = discriminator_template() 
    g_template = generator_template()

    gen_images, z_prediction = pt.construct_all(g_template, input=z)
    tf.image_summary('generated_images', gen_images, max_images=FLAGS.batch_size, name='generated_images_summary')

    real_logits = d_template.construct(input=real_images)
    fake_logits = d_template.construct(input=gen_images)

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_logits, tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logits, tf.zeros_like(fake_logits)))
    discriminator_loss = tf.add(real_loss, fake_loss, name='discriminator_loss')

    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logits, tf.ones_like(fake_logits)), name='generator_loss')

    z_prediction_loss = tf.reduce_mean(tf.square(z - z_prediction), name='z_prediction_loss')

    tf.add_to_collection('losses', generator_loss)
    tf.add_to_collection('losses', discriminator_loss)
    tf.add_to_collection('losses', z_prediction_loss)

    return generator_loss, discriminator_loss, z_prediction_loss

def train(loss, global_step, net=None):
    if net == 'generator':
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        opt = gen_optimizer()
    elif net == 'discriminator':
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        opt = discrim_optimizer()
    elif net == 'z_predictor':
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        opt = gen_optimizer()
    else:
        raise RuntimeError('Net to train must be one of generator, discriminator, or z_predictor.')

    # Compute gradients.
    grads = opt.compute_gradients(loss, var_list=variables)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step, name='train_'+net)

    # Add histograms for trainable variables.
    #for var in tf.trainable_variables():
    #    tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    #for grad, var in discrim_grads + gen_grads:
    #    if grad is not None:
    #        tf.histogram_summary(var.op.name + '/gradients', grad)

    return apply_gradient_op

