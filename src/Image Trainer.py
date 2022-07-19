import numpy as np
from PIL import Image

def innit_image():
    image = Image.open("testImages/test.png")
    image = np.array(image)
    image = np.argmax(image, 2)
    image = np.minimum(image, 1)
    printImage(image)
    image = np.reshape(image, 784)
    return(image)

def innit_params():
    W1 = np.load('saves/W1.npy')
    W2 = np.load('saves/W2.npy')
    b1 = np.load('saves/b1.npy')
    b2 = np.load('saves/b2.npy') 
    return W1, W2, b1, b2

def update_files(W1, W2, b1, b2):
    return

def ReLu(Z):
    return np.maximum(Z, 0)

def deriv_ReLu(Z):
    return Z > 0

def SoftMax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def one_hot(Y):
    one_hot_Y = np.zeros((10,1))
    one_hot_Y[Y] = 1
    return one_hot_Y

def foward_prop(W1, W2, b1, b2, X):
    Z1 = W1.dot(X) + b1.T
    A1 = ReLu(Z1.T)

    Z2 = W2.dot(A1) + b2
    A2 = SoftMax(Z2)
    return Z1, Z2, A1, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    X = np.reshape(X,(784,1))

    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = dZ2.dot(A1.T)
    db2 = np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLu(Z1).T
    dW1 = (X.dot(dZ1.T)).T
    db1 = np.sum(dZ1)
    return dW1, dW2, db1, db2

def update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, W2, b1, b2

def printImage(image):
    image = image.tolist()
    for i in range(len(image)):
        for l in range(len(image)):
            if image[i][l] == 0:
                image[i][l] = "."
            else:
                image[i][l] = "#"
        print(image[i])

def save(W1, W2, b1, b2):
    np.save('saves/W1.npy', W1)
    np.save('saves/b1.npy', b1)
    np.save('saves/W2.npy', W2)
    np.save('saves/b2.npy', b2)

image = innit_image()
W1, W2, b1, b2 = innit_params()
Z1, Z2, A1, A2 = foward_prop(W1, W2, b1, b2, image)

print(np.argmax(A2,0))
answer = int(input("The draw number? \n"))

dW1, dW2, db1, db2 = back_prop(Z1, A1, Z2, A2, W2, image, answer)

print(W1.shape , W2.shape , b1.shape , b2.shape)

W1, W2, b1, b2 = update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, 0.1)

save(W1, W2, b1, b2)