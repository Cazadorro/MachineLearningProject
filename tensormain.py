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
learning_rate = 0.01
epochs = 5
batch_size = 128

def t_main():
    test_x = read_csv("map2_x_data.csv", float)
    test_y = read_csv("map2_y_data.csv", float)
    input_x = read_csv("map1_x_data.csv", float)
    expected_y = read_csv("map1_y_data.csv", float)
    # input rays and goal + start
    x = tf.placeholder(tf.float32, shape=[None, input_num_units])
    y = tf.placeholder(tf.float32, shape= [None, output_num_units])
    W = tf.Variable(tf.zeros([]))
    weights = {
        'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed1)),
        'hidden2': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed1)),
        'hidden3': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed1)),
        'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed2))
    }
    biases = {
        'hidden1': tf.Variable(tf.random_normal([hidden_num_units], seed=seed1)),
        'hidden2': tf.Variable(tf.random_normal([hidden_num_units], seed=seed1)),
        'hidden3': tf.Variable(tf.random_normal([hidden_num_units], seed=seed1)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed2))
    }
    hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)
    hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    hidden_layer3 = tf.add(tf.matmul(hidden_layer2, weights['hidden3']), biases['hidden3'])
    hidden_layer3 = tf.nn.relu(hidden_layer3)
    output_layer = tf.matmul(hidden_layer3, weights['output']) + biases['output']
    #output_layer = tf.nn.relu(output_layer)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
    print(output_layer.get_shape())
    print(y.get_shape())

    cost1 = tf.reduce_mean(tf.abs(tf.subtract(output_layer, y))[0])
    cost2 = tf.reduce_mean(tf.abs(tf.subtract(output_layer, y))[1])
    cost = tf.add(cost1, cost2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    split_v = 100

    with tf.Session() as sess:
        # create initialized variables
        writer = tf.train.SummaryWriter('tf_logs', tf.get_default_graph())
        sess.run(init)

        ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize
        for step in range(10):
            _, c = sess.run([optimizer, cost], feed_dict={x: input_x[:], y: expected_y[:]})
            #tf.Print(output_layer)
            print('cost ', c)


        print("\nTraining complete!")

        # find predictions on val set
        pred_temp1 = tf.less_equal(tf.abs(tf.subtract(output_layer, y))[0],  0.5)
        pred_temp2 = tf.less_equal(tf.abs(tf.subtract(output_layer, y))[1], 0.5)
        pred_temp = pred_temp1 #tf.add(pred_temp1, pred_temp2)
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("asdf")
        print(sess.run(accuracy, feed_dict={x: input_x[:split_v], y: expected_y[:split_v]}))
        #print("Validation Accuracy:", accuracy.eval({x: input_x[:split_v], y: expected_y[:split_v]}))

        predict = output_layer
        pred = predict.eval({x: test_x[1].reshape(-1, 20)})
        print(pred)
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
