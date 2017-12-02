# Bitcoin Price Predictor - Linear Regression w/ 1 Feature
# 11/17/17

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

# Set Parameters
learning_rate = 0.3
training_epochs = 20000
display_step = 500
random = np.random
n = 4  # Polynomial Power

# Get features
_, Y = data.get_data("BCHAIN/MKPRU", 365)
_, X = data.get_data("BCHAIN/BLCHS", 365)

# Number of training examples
m = Y.shape[0]

# Mean Normalization and Feature Scaling
train_X = np.zeros([n, m])
for i in range(n):
    X_temp = np.power(X, i+1)
    train_X[i, :] = (X_temp - np.mean(X_temp))/np.std(X_temp, axis=0)



# Set model weight and bias
theta = tf.Variable(tf.cast(tf.random_uniform([1, n], -1.0, 1.0), tf.float64), name="theta")
bias = tf.Variable(tf.cast(tf.random_uniform([1], -1.0, 1.0), tf.float64), name="bias")

# Hypothesis of Market Price
hypothesis = tf.matmul(theta, train_X) + bias

# Mean Squared Error
cost = tf.reduce_mean(tf.square(hypothesis - Y)) / (2 * m)

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

    # Display training results
    plt.plot(X, Y, 'r', label='Market Price (training)')
    plt.plot(X, np.transpose(S.run(hypothesis)), label='Polynomial Regression Line')
    plt.title("Bitcoin Price Prediction: Training Polynomial Regression")
    plt.xlabel("BlockChain Size")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()

    test_X = np.array([100000, 150000])
    test_X = (test_X - np.mean(X))/np.std(X, axis=0)


    print("Given Blockchain Size = 100,000 -> Hypothesis = ", test_X[0]*S.run(theta) + S.run(bias))
    print("Given Blockchain Size = 150,000 -> Hypothesis = ", test_X[1]*S.run(theta) + S.run(bias))

