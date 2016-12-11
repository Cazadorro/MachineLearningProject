#!/bin/bash

import tensorflow as tf
from src.testgeneration import read_csv
import numpy as np

NUM_RAYS = 16
input_num_units = NUM_RAYS + 4
hidden_num_units = input_num_units
output_num_units = 2
seed1 = 12334234
seed2 = 58764521124
seed1 = seed2 = None
learning_rate = 0.00001
epochs = 5
batch_size = 50
train_size = 100

def init_weights(shape, init_method='xavier', xavier_params = (None, None), stddev = 0.01, minval=0, maxval=None):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'normal':
        return tf.Variable(tf.random_normal(shape, stddev=stddev, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_uniform(shape, minval=minval, maxval=maxval, dtype=tf.float32))
    elif init_method == 'xavier':
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    elif init_method == 'xavieri':
        return tf.Variable(tf.random_uniform(shape, initializer=tf.contrib.layers.xavier_initializer()))


def model(X, num_hidden=10):
    input_size = input_num_units
    output_size = output_num_units
    w_h = init_weights([input_size, num_hidden], 'xavier', xavier_params=(input_size, num_hidden))
    b_h = init_weights([num_hidden], 'zeros')
    h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)

    w_o = init_weights([num_hidden, output_size], 'xavier', xavier_params=(num_hidden, output_size))
    b_o = init_weights([output_size], 'zeros')
    return tf.matmul(h, w_o) + b_o


def t_main():
    test_x = read_csv("map2_x_data.csv", float)
    test_y = read_csv("map2_y_data.csv", float)
    input_x = read_csv("map1_x_data.csv", float)
    expected_y = read_csv("map1_y_data.csv", float)
    train_x = input_x[:train_size]
    train_y = expected_y[:train_size]
    test_x = input_x[train_size:]
    test_y = expected_y[train_size:]

    # input rays and goal + start
    x = tf.placeholder(tf.float32, shape=[None, input_num_units])
    y = tf.placeholder(tf.float32, shape= [None, output_num_units])

    yhat = model(x, 1)

    cost = tf.abs(tf.subtract(yhat, y))
    #cost = tf.nn.l2_loss(yhat - y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    split_v = 100

    with tf.Session() as sess:
        # create initialized variables
        writer = tf.train.SummaryWriter('tf_logs', tf.get_default_graph())
        sess.run(init)
        errors = []
        for i in range(epochs):
            for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x), batch_size)):
                sess.run(optimizer, feed_dict={x: train_x[start:end], y: train_y[start:end]})
            mse = sess.run(tf.nn.l2_loss(yhat - test_y), feed_dict={x: test_x})
            errors.append(mse)
            print("epoch %d, validation MSE %g" % (i, mse))
            if i % 100 == 0:
                print("epoch %d, validation MSE %g" % (i, mse))

        merged = tf.summary.merge_all()



def accuracy(prediction, result):
    pass


if __name__ == "__main__":
    # run main program
    t_main()
