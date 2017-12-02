# Bitcoin Price Predictor - Linear Regression w/ 1 Feature
# 11/17/17

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

# Set Parameters
learning_rate = 0.3
training_epochs = 10000
display_step = 500
random = np.random

# Get features
X, Y = data.get_data("BCHAIN/MKPRU", 365)
_, X1 = data.get_data("BCHAIN/BLCHS", 365)



# Mean Normalization and Feature Scaling
train_X = (X - np.mean(X))/np.std(X, axis=0)


# Number of training examples
m = len(train_X)

# Set model weight and bias
theta = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="theta")
bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="bias")

# Hypothesis of Market Price
hypothesis = train_X*theta + bias

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
    plt.plot(X, S.run(hypothesis), label='Linear Regression Line')
    plt.title("Bitcoin Price Prediction: Training Linear Regression")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()






    test_X = np.array([100000, 150000])
    test_X = (test_X - np.mean(X))/np.std(X, axis=0)


    print("Given Blockchain Size = 100,000 -> Hypothesis = ", test_X[0]*S.run(theta) + S.run(bias))
    print("Given Blockchain Size = 150,000 -> Hypothesis = ", test_X[1]*S.run(theta) + S.run(bias))

