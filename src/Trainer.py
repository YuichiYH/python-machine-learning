import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/mnist_train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255 #255 means how white the pixel are, and making the value go below 1

def ReLu(Z):
    return np.maximum(Z, 0)
    
def deriv_ReLu(Z):
    return Z > 0

def SoftMax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1)- 0.5

    W2 = np.random.rand(10, 10)- 0.5
    b2 = np.random.rand(10, 1)- 0.5
    
    return(W1, b1, W2, b2)

def foward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = SoftMax(Z2)

    return(Z1, A1, Z2, A2)

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return(dW1, db1, dW2, db2)

def update_parans(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return(W1, b1, W2, b2)

def get_prediction(A2):
    return(np.argmax(A2, 0))

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return(np.sum(predictions == Y) / Y.size)

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = foward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parans(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("iteration: ", i)
            print("Accuracy: ", get_accuracy(get_prediction(A2), Y))
    return(W1, b1, W2, b2)

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)

np.save('saves/W1.npy', W1)
np.save('saves/b1.npy', b1)
np.save('saves/W2.npy', W2)
np.save('saves/b2.npy', b2)