#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from scipy import stats
import pickle
import sys

channels = {
    'input': 3,
    'conv1': 60,
    'conv2': 6,
    'fc': 1000,
}

fc_in_width = 1
fc_in_height = 1
fc_in_depth = channels['input'] * channels['conv1'] * channels['conv2']

input_height = 1
input_width = 90
n_labels = 6  # Downstairs, Jogging, Sitting, Standing, Upstairs, Walking
kernel_size = 60

DATASET_FILE = os.path.join('WISDM_ar_v1.1', 'WISDM_ar_v1.1_raw.txt')
PICKLE_FNAME = 'WISDM_ar_v1.1.pickle'

column_names = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']


def HarCnn(x, weights, biases, net={}):
    """
    CNN model

    x: [1 x 90 x 3] -- (conv, ksize=60, n_chs=3, depth=60, strides=1) --> [1 x 31 x 180] -- (pool, ksize=20, strides=2) --> [1 x 6 x 180]
    -- (conv, ksize=6, n_chs=3*60, depth=6, strides=1) --> [1 x 1 x (180*6)] -- FC --> [1080] -- Softmax --> [6]
    """
    with tf.name_scope('conv1') as scope:
        conv1 = tf.nn.depthwise_conv2d(
            x, weights['conv1'], [1, 1, 1, 1], padding='VALID')
        conv1 = tf.nn.bias_add(conv1, biases['conv1'])
        conv1 = tf.nn.relu(conv1)
    net['conv1'] = conv1

    pool = tf.nn.max_pool(
        conv1, ksize=[1, 1, 20, 1], strides=[1, 1, 2, 1], padding='VALID')
    net['pool'] = pool

    with tf.name_scope('conv2') as scope:
        conv2 = tf.nn.depthwise_conv2d(
            pool, weights['conv2'], [1, 1, 1, 1], padding='VALID')
        conv2 = tf.nn.bias_add(conv2, biases['conv2'])
        conv2 = tf.nn.relu(conv2)
    net['conv2'] = conv2

    shape = conv2.get_shape().as_list()
    with tf.name_scope('fc') as scope:
        fc = tf.reshape(conv2, shape=[-1, shape[1] * shape[2] * shape[3]])
        fc = tf.nn.bias_add(tf.matmul(fc, weights['fc']), biases['fc'])
        fc = tf.nn.tanh(fc)
    net['fc'] = fc

    with tf.name_scope('out') as scope:
        out = tf.nn.bias_add(tf.matmul(fc, weights['out']), biases['out'])
        out = tf.nn.softmax(out)
    net['out'] = out

    return out


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)


def segment_signal(data, fname, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        if x.shape[0] == window_size and \
           y.shape[0] == window_size and \
           z.shape[0] == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels,
                               stats.mode(data['activity'][start:end])[0][0])
    # save segments and labels as a pickle file
    with open(fname, 'wb') as f:
        pickle.dump([segments, labels], f)
    return segments, labels


def read_segments_and_labels(fname):
    with open(fname, 'rb') as f:
        segments, labels = pickle.load(f)
    return segments, labels


def main():
    if len(sys.argv) > 1:
        # read dataset from a pickle file
        segments, labels = read_segments_and_labels(sys.argv[1])
    else:
        # generate new dataset and save as a pickle file for future use.
        fname = PICKLE_FNAME
        dataset = pd.read_csv(DATASET_FILE, header=None, names=column_names)
        segments, labels = segment_signal(dataset, fname)

    x = tf.placeholder(tf.float32, [
        None,
        input_height,
        input_width,
        channels['input'],
    ])
    y = tf.placeholder(tf.float32, [None, n_labels])

    # load dataset
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    reshaped_segments = segments.reshape(
        len(segments), 1, input_width, channels['input'])

    # train and test set
    train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
    train_x = reshaped_segments[train_test_split]
    train_y = labels[train_test_split]
    test_x = reshaped_segments[~train_test_split]
    test_y = labels[~train_test_split]

    # weights and biases
    weights = {
        'conv1':
        tf.Variable(
            tf.truncated_normal(
                [1, kernel_size, channels['input'], channels['conv1']],
                stddev=0.1)),
        'conv2':
        tf.Variable(
            tf.truncated_normal(
                [
                    1, 6, channels['input'] * channels['conv1'], channels[
                        'conv2']
                ],
                stddev=0.1)),
        'fc':
        tf.Variable(
            tf.truncated_normal(
                [fc_in_width * fc_in_height * fc_in_depth, channels['fc']],
                stddev=0.1)),
        'out':
        tf.Variable(
            tf.truncated_normal([channels['fc'], n_labels], stddev=0.1)),
    }
    biases = {
        'conv1':
        tf.Variable(
            tf.constant(0.0, shape=[channels['input'] * channels['conv1']])),
        'conv2':
        tf.Variable(
            tf.constant(
                0.0,
                shape=[
                    channels['input'] * channels['conv1'] * channels['conv2']
                ])),
        'fc':
        tf.Variable(tf.constant(0.0, shape=[channels['fc']])),
        'out':
        tf.Variable(tf.constant(0.0, shape=[n_labels])),
    }

    # construct a model
    net = {}
    pred = HarCnn(x, weights, biases, net)

    # loss
    learning_rate = 0.001
    cross_entropy = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # initialize all variable
    init = tf.global_variables_initializer()

    # launch a model
    training_epochs = 5
    batch_size = 10
    total_batch = train_x.shape[0] // batch_size
    display_step = 1
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0
            for b in range(total_batch):
                offset = (b * batch_size) % (total_batch)
                batch_x = train_x[offset:(offset + batch_size), :, :, :]
                batch_y = train_y[offset:(offset + batch_size), :]
                _, c = sess.run(
                    [train_step, cross_entropy],
                    feed_dict={x: batch_x,
                               y: batch_y})
                avg_cost += (c / total_batch)
            if epoch % display_step == 0:
                print("Epoch %04d" % (epoch + 1),
                      "cost = {:.9f}".format(avg_cost))
        print('Training finished.')

        # eval for the test set
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy: ', accuracy.eval({x: test_x, y: test_y}))


if __name__ == '__main__':
    main()
