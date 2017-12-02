# Bitcoin Price Predictor - Linear Regression
# 11/17/17

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

# Set Parameters
learning_rate = .3
training_epochs = 10000
display_step = 500

# List of all Features Used
features = ["BCHAIN/MWNUS", "BCHAIN/BLCHS"]
n = len(features)

# Get Y data
_, train_Y = data.get_data("BCHAIN/MKPRU", 365)
m = train_Y.shape[0]  # Number of training examples

# Get X data, perform Mean Normalization and Feature Scaling
X_data = np.zeros([n, m]) # un-normalized
train_X = np.zeros([n, m])
for i, feature in enumerate(features):
    _, X = data.get_data(feature, 365)
    train_X[i, :] = (X - np.mean(X)) / np.std(X, axis=0)
    X_data[i, :] = X

# Set model weight and bias
theta = tf.Variable(tf.cast(tf.random_uniform([1, n], -1.0, 1.0), tf.float64), name="theta")
bias = tf.Variable(tf.cast(tf.random_uniform([1], -1.0, 1.0), tf.float64), name="bias")

# Hypothesis
hypothesis = tf.matmul(theta, train_X) + bias

# Mean Squared Error
cost = tf.reduce_mean(tf.square(hypothesis - train_Y)) / (2 * m)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Start Tensorflow session
with tf.Session() as S:

    # Initializer
    S.run(tf.global_variables_initializer())

    # Training data
    for epoch in range(training_epochs):
        S.run(optimizer)

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = S.run(cost)
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "theta=", S.run(theta), "bias=", S.run(bias))

    training_cost = S.run(cost)
    print("Training cost=", training_cost, "theta=", S.run(theta), "bias=", S.run(bias), '\n')


    # Running example, 12,000,000 wallets, and Blockchain Size of 100,000
    test_X = np.reshape([12000000.0, 100000.0], (2, 1))
    test_X[0] = (test_X[0] - np.mean(X_data[0]))/np.max(X_data[0], axis=0)
    test_X[1] = (test_X[1] - np.mean(X_data[1]))/np.max(X_data[1], axis=0)
    print(S.run(tf.matmul(theta, test_X) + bias))




