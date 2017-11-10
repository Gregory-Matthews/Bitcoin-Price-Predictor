import quandl
import tensorflow as tf
import matplotlib.pyplot as mplot
import numpy as np

def main():
    data = quandl.get("BCHAIN/MKPRU", returns="numpy")
    m = data.shape[0] # number of training examples
    print data.shape

    i = 0
    Y = np.empty([m,1])
    X = np.arange(0, m)
    for y in data:
        Y[i] = y[1]
        i+=1
        
    print X
    print Y

    #X = tf.placeholder("float")
    #Y = tf.placeholder("float")

    mplot.plot(X, Y, 'ro', label='Bitcoin Market Price')
    mplot.legend()
    mplot.show()


if __name__ == "__main__":
    main()



