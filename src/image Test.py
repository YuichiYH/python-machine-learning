import numpy as np
from PIL import Image

image = Image.open("testImages/test.png")
# image.show()
# data = pd.read_csv('data/mnist_test.csv')

def init_params():
    W1 = np.load('saves/W1.npy') # 10 x 784
    W2 = np.load('saves/W2.npy') # 10 x 1
    b1 = np.load('saves/b1.npy') # 10 x 10
    b2 = np.load('saves/b2.npy') # 10 x 1
    return W1, b1, W2, b2

def ReLu(Z):
    return np.maximum(Z, 0)

def SoftMax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def test(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1.T
   
    A1 = ReLu(Z1.T)

    Z2 = W2.dot(A1) + b2
    A2 = SoftMax(Z2)

    output = np.argmax(A2,0)
    return(output[0])

def accuracity_Test(array, n):
    """
    Tests the accuracity of the Neural Netword
    Arguments:
        array = the mnist_test.csv file
        n = number of tests
    Returns:
        The accuracity of the test
    """
    W1, b1, W2, b2 = init_params()
    
    accuracity = 0

    for i in range(n):
        data = array[i]
        answer = data[0]
        image = data[1:].T / 255
        
        output = test(W1, b1, W2, b2, image)

        if answer == output:
            accuracity += 1
        # else:
        #     print("iteration:", i, answer, output)

        
        if i % 10 == 0:
            print((accuracity/(i+1))*100)

def printImage(image):
    """
    Prints the list as an image
    """

    for i in range(len(image)):
        for l in range(len(image)):
            if image[i][l] == 0:
                image[i][l] = "."
            else:
                image[i][l] = "#"
        print(image[i])

# data = np.array(data)

# accuracity_Test(data, 100)
W1, b1, W2, b2 = init_params()

image = np.array(image)
image = np.argmax(image, 2)
image = np.minimum(image, 1)

printImage(image.tolist())

image = np.reshape(image, 784)

print(test(W1, b1, W2, b2, image))