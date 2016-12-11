#!/bin/bash

import tensorflow as tf
from src.testgeneration import read_csv
import numpy as np
import random

NUM_RAYS = 16
input_num_units = NUM_RAYS + 4
hidden_num_units = input_num_units
output_num_units = 2
seed1 = 12334234
seed2 = 58764521124
seed1 = seed2 = None
learning_rate = .01
epochs = 5
batch_size = 10

minv = 0
maxv = 1

hidden_string = 'hidden_layer_{}'
output_string = 'output'
typeused = tf.float32

def gen_weights(num_hidden, init_method):
    weights = {}
    if num_hidden > 0:
        output_before_size = hidden_num_units
        weights[hidden_string.format(0)] = tf.Variable(init_method([input_num_units, hidden_num_units]),
                                                       dtype=typeused)
    else:
        output_before_size = input_num_units
    for layer_number in range(1, num_hidden):
        weights[hidden_string.format(layer_number)] = tf.Variable(init_method([hidden_num_units, hidden_num_units]),
                                                                  dtype=typeused)
    weights[output_string] = tf.Variable(init_method([output_before_size, output_num_units]), dtype=typeused)
    return weights


def gen_biases(num_hidden, init_method):
    biases = {}
    if num_hidden > 0:
        biases[hidden_string.format(0)] = tf.Variable(init_method([hidden_num_units]), dtype=typeused)
    for layer_number in range(1, num_hidden):
        biases[hidden_string.format(layer_number)] = tf.Variable(init_method([hidden_num_units]), dtype=typeused)
    biases[output_string] = tf.Variable(init_method([output_num_units]), dtype=typeused)
    return biases


def gen_network(placeholder_x, keep_prob, num_hidden, init_method, activation_function):
    weights = gen_weights(num_hidden, init_method)
    biases = gen_biases(num_hidden, init_method)
    if num_hidden > 0:
        last_layer = tf.add(tf.matmul(placeholder_x, weights[hidden_string.format(0)]), biases[hidden_string.format(0)])
        last_layer = activation_function(last_layer)
    else:
        last_layer = placeholder_x
    for layer_number in range(1, num_hidden):
        last_layer = tf.add(tf.matmul(last_layer, weights[hidden_string.format(layer_number)]),
                            biases[hidden_string.format(layer_number)])
        last_layer = activation_function(last_layer)
    last_layer = tf.nn.dropout(last_layer, keep_prob)
    output_layer = tf.nn.l2_normalize(tf.matmul(last_layer, weights[output_string]) + biases[output_string], 1)
    return output_layer


def t_main():
    test_x = read_csv("map2_x_data.csv", float)
    test_y = read_csv("map2_y_data.csv", float)
    input_x = read_csv("map1_x_data.csv", float)
    expected_y = read_csv("map1_y_data.csv", float)
    # input rays and goal + start
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, input_num_units])
    y = tf.placeholder(tf.float32, shape=[None, output_num_units])
    init_method = tf.random_normal
    #init_method = tf.zeros
    #init_method = tf.random_uniform
    # minval=minv, maxval=maxv,
    #relu
    activation_function = tf.nn.sigmoid
    output_layer = gen_network(x, keep_prob, 3, init_method, activation_function)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
    print(output_layer.get_shape())
    print(y.get_shape())

    cost1 = tf.reduce_mean(tf.abs(tf.subtract(output_layer, y))[0])
    cost2 = tf.reduce_mean(tf.abs(tf.subtract(output_layer, y))[1])
    # cost = tf.abs(tf.subtract(output_layer, y))
    cost = tf.nn.l2_loss(output_layer - y)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    split_v = 100

    with tf.Session() as sess:
        # create initialized variables
        writer = tf.train.SummaryWriter('tf_logs', tf.get_default_graph())
        sess.run(init)

        epochs = int(len(input_x) / batch_size)
        l_input = list(input_x)
        l_expect = list(expected_y)
        overfit_iters = 2
        for step in range(overfit_iters):
            for i in range(epochs):
                start = i * batch_size
                end = start + batch_size
                _, c = sess.run([optimizer, cost], feed_dict={x: input_x[start:end], y: expected_y[start:end], keep_prob: 0.5})
                #_, c = sess.run([optimizer, cost], feed_dict={x: random.sample(l_input, batch_size), y: random.sample(l_expect, batch_size)})

                if i % 100 == 0:
                    print(c)
                    # tf.Print(output_layer)

        print("\nTraining complete!")

        # find predictions on val set
        pred_temp = tf.less_equal(tf.abs(output_layer - y), [0.2, 0.2])
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("accuracy")
        print(sess.run(accuracy, feed_dict={x: input_x[:], y: expected_y[:], keep_prob: 1.0}))
        # print("Validation Accuracy:", accuracy.eval({x: input_x[:split_v], y: expected_y[:split_v]}))

        predict = output_layer
        pred = predict.eval({x: test_x[1].reshape(-1, 20), keep_prob: 1.0})
        print(pred)
        print(test_y[1])
        merged = tf.summary.merge_all()


def accuracy(prediction, result):
    pass


def main():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    with tf.Session() as sess:
        result = sess.run(product)
        print(result)
    pass


if __name__ == "__main__":
    # run main program
    t_main()
