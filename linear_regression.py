# Bitcoin Price Predictor - Linear Regression
# 11/17/17

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

# Set Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
random = np.random

# Get features
train_X, train_Y = data.get_data("BCHAIN/MKPRU", 1)

test_X, test_Y = train_X[-train_X.shape[0]/2:], train_Y[-train_Y.shape[0]/2:]
train_X, train_Y = train_X[0:train_X.shape[0]/2], train_Y[0:train_Y.shape[0]/2]

# Set inout and output variables
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weight and bias
theta = tf.Variable(random.randn(), name="weight")
bias = tf.Variable(random.randn(), name="bias")

# Mean Squared Error
hypothesis = tf.add(tf.multiply(X, theta), bias)
cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / (2 * train_X.shape[0])

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Start Tensorflow session
with tf.Session() as S:

    # Initializer
    S.run(tf.global_variables_initializer())

    # Training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            S.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = S.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "theta=", S.run(theta), "bias=", S.run(bias))

    training_cost = S.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", S.run(theta), "b=", S.run(bias), '\n')

    # Display training results
    plt.plot(train_X, train_Y, 'r', label='Market Price (training)')
    plt.plot(train_X, S.run(theta) * train_X + S.run(bias), label='Linear Regression Line')
    plt.title("Bitcoin Price Prediction: Training Linear Regression")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()

    # Display testing results
    plt.plot(test_X, test_Y, 'b', label='Market Price (testing)')
    plt.plot(test_X, S.run(theta) * test_X + S.run(bias), label='Linear Regression Line')
    plt.title("Bitcoin Price Prediction: Testing Linear Regression")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()
