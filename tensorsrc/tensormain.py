#!/bin/bash

import tensorflow as tf

NUM_RAYS = 16
input_num_units = NUM_RAYS + 4
hidden_num_units = input_num_units
output_num_units = 2
seed1 = 12334234
seed2 = 58764521124
learning_rate = 0.001
epochs = 5
batch_size = 128

def t_main():
    # input rays and goal + start
    x = tf.placeholder(tf.float32, shape=[None, input_num_units])
    y = tf.placeholder(tf.float32, shape= [None, output_num_units])
    W = tf.Variable(tf.zeros([]))
    weights = {
        'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed1)),
        'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed2))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed1)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed2))
    }
    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        # create initialized variables
        sess.run(init)

        ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize

        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(train.shape[0] / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                avg_cost += c / total_batch

            print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

        print("\nTraining complete!")

        # find predictions on val set
        pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, 784), y: dense_to_one_hot(val_y)}))

        predict = tf.argmax(output_layer, 1)
        pred = predict.eval({x: test_x.reshape(-1, 784)})

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
    main()
