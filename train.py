from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import sys
import os.path
import time

import numpy as np
import tensorflow as tf

import model
import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './tmp/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_float('learning_rate', 0.0002,
                          'Learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Batch size.')
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_integer('discriminator_steps', 1,
                            '''Number of steps to train the discriminator.''')
tf.app.flags.DEFINE_integer('generator_steps', 1,
                            '''Number of steps to train the generator.''')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            '''Whether to log device placement.''')
tf.app.flags.DEFINE_boolean('restore_model', False,
                            '''Whether to restore model from the checkpoint directory.''')
tf.app.flags.DEFINE_float('keep_checkpoint_every_n_hours', 10000.0,
                          'Regularly save and keep checkpoint files.')
tf.app.flags.DEFINE_boolean('save_discriminator', False,
                            'Whether to save the discriminator variables in addition to the generator.')

def init_variables(sess):
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    variables_to_restore = gen_vars + discrim_vars
    saver = tf.train.Saver(variables_to_restore)
    if FLAGS.restore_model:
        ckpt = tf.train.get_checkpoint_state('./')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('No checkpoint file found.')
    return

def train():
    input.init_dataset_constants()
    # build model into default graph
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # get images
        with tf.variable_scope('input_pipeline') as scope:
            images = input.inputs()

        train_ops = []
        # calculate loss
        gen_loss, discrim_loss, z_prediction_loss = model.losses(images)

        # loss summaries
        for l in tf.get_collection('losses'):
            tf.scalar_summary(l.op.name, l)

        # build graph that trains the model with one batch of examples
        gen_train_op = model.train(gen_loss, global_step, net='generator')
        discrim_train_op = model.train(discrim_loss, global_step, net='discriminator')
        z_predictor_train_op = model.train(z_prediction_loss, global_step, net='z_predictor')

        # create a saver
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        variables_to_save = gen_vars
        if FLAGS.save_discriminator:
            variables_to_save += discrim_vars
        saver = tf.train.Saver(variables_to_save, max_to_keep=1,
                               keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

        # build the summary operation based on the TF collection of summaries
        summary_op = tf.merge_all_summaries()

        # start running operations on the graph
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        # initialize
        init_variables(sess)

        # start the queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        net = 'discriminator'
        count = 0

        for step in xrange(FLAGS.max_steps):
            if net == 'discriminator':
                op, loss = discrim_train_op, discrim_loss
                count += 1
                if count == FLAGS.discriminator_steps:
                    net = 'generator'
                    count = 0
            else:
                op, loss = gen_train_op, gen_loss
                count += 1
                if count == FLAGS.generator_steps:
                    net = 'discriminator'
                    count = 0

            start_time = time.time()

            _, loss_value = sess.run([op, loss])
            assert not np.isnan(loss_value), 'Model diverged with NaN loss value'
            if net == 'generator':
                _, loss_value = sess.run([z_predictor_train_op, z_prediction_loss])
                assert not np.isnan(loss_value), 'Model diverged with NaN loss value'

            duration = time.time() - start_time

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = '{}: step {}, ({:.0f} examples/sec; {:.3f} sec/batch)'
                print(format_str.format(datetime.now(), step, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # save the model checkpoint periodically
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = FLAGS.train_dir
                saver.save(sess, checkpoint_path, global_step=step)

        # ask threads to stop
        coord.request_stop()

        # wait for threads to finish
        coord.join(threads)
        sess.close()

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    
    train()

if __name__ == '__main__':
    tf.app.run()

