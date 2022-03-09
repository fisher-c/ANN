#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# hw6_ann.py
# HW6: Artificial Neural Networks
# Comp 131 AI
# Carina Ye
# Dec 17, 2021
# Implements a multi-layer neural network to classify iris plants.
"""

# import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize 

np.random.seed(42)

#import dataset
iris = pd.read_csv("ANN - Iris data.txt",  header = None)

# select all rows and first four columns as feature data
# Convert data to numpy array in order for processing
X = iris.iloc[:, 0:4].values
y_class = iris.iloc[:, -1].values


"""
# change iris labels to one hot vector
# [0]--->[1 0 0]
# [1]--->[0 1 0]
# [2]--->[0 0 1]
"""
y = OneHotEncoder(sparse = False).fit_transform(y_class.reshape(-1, 1))

# normalization: normalize features to a range of 0-1 for processing
#X_normalized = normalize(X, axis=0)

# split into test and train sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# convert input into matrix form
x_train = x_train.T
x_test = x_test.T  
y_train = y_train.T
y_test = y_test.T

 
"""
initialize weights and biases for the Neural Net
2 layers: 1 hidden layer (4 neurons), ouput layer (3 neurons)
W1: layer1 weight
W2: layer2 weight
b1: layer1 bias
b2: layer2 bias
"""
def init_params():
    W1 = np.random.rand(4, 4) 
    b1 = np.random.rand(4, 1) 
    W2 = np.random.rand(3, 4) 
    b2 = np.random.rand(3, 1) 
    return W1, b1, W2, b2


# ReLU activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

#derivative of ReLU
def ReLU_deriv(Z):
    return Z > 0

# sigmoid activation function
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


"""
forward propagation to calculate A2 (probability)
A1: activation for layer 1
A2: activation for layer 2
"""
def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1        # order matters!
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2   
    A2 = softmax(Z2)
    #print("A2:", A2)
    return Z1, A1, Z2, A2


#Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x_train)

"""
backward propagation process
"""
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]  #number of training examples
    dZ2 = A2 - Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1)
    return dW1, db1, dW2, db2

#dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, x_train, y_train)


"""
updates weights and biases after a round of forward and backward prop
alpha: learning rate
"""
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


"""
predicts iris class
"""
def get_predictions(A2):
    return np.argmax(A2, 0)


"returns ANN accuracy (number of correct predictions)"
def get_accuracy(predictions, Y):
    #transform our predictions using one hot encoder so that its shape match Y
    predictions = OneHotEncoder(sparse = False).fit_transform(predictions.reshape(-1, 1))
    predictions = predictions.T
    #print("predictions: ", predictions, "\n\nActuals: ", Y)
    #print(predictions.shape, Y.shape)
    return (np.sum(predictions == Y)) / Y.size



"""
Main process of training a Neural Net:
initializes a neural network, then uses forward propagation, 
backward propagation, updates parameters, and make predictions
"""
def NeuralNet(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    #gradient descent
    for i in range(iterations + 1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 100 == 0:   # print performance every 50 iterations
            print("\nIteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: %.4f" % get_accuracy(predictions, Y))
    return W1, b1, W2, b2



def make_predictions(X, W1, b1, W2, b2):
    W1, b1, W2, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    #predictions = OneHotEncoder(sparse = False).fit_transform(predictions.reshape(-1, 1))
    #predictions = predictions.T
    #print("here", predictions)
    return predictions


#given an input iris, print out its predicted class
def test_prediction(X, W1, b1, W2, b2):
    prediction = make_predictions(X, W1, b1, W2, b2)
    genre = "NA"
    if prediction[0] == 0:
        genre = "Iris Setosa"
    elif prediction[0] == 1:
        genre = "Iris Versicolour"
    elif prediction[0] == 2:
        genre = "Iris Virginica"    
    print("Your input iris is predicted to be in class: ", genre)    
    
    

#main: prompt user input, predict iris class
if __name__=="__main__": 
    #look at the ANN performance
    print('-----------------TESTING ANN-------------------')     
    print("Using a 0.3 test size, the performance of our ANN is: ")
    # fitting NN on training and testing sets to see the performance
    print("Performance on Training Set:")
    W1, b1, W2, b2 = NeuralNet(x_train, y_train, 0.10, 500)
    print("----------------------------------\nPerformance on Testing Set:")
    W11, b11, W21, b21 = NeuralNet(x_test, y_test, 0.10, 500)
    # ask for user inputs
    while True:
        print('--------------------QUERY----------------------')     
        str_arr = input("Plese enter iris attributes, "\
                         "separated by commas (e.g. 5.1, 3.5, 1.4, 0.2): "\
                        ).split(',')
        print("\nYour input iris attributes are: ", str_arr)
        
        int_arr = [float(x) for x in str_arr]
        iris_arr = np.reshape(int_arr, (-1, 1))
        test_prediction(iris_arr, W1, b1, W2, b2)
        print('---------------------------------')     
        var = input("Enter anything to continue, enter 'q' to quit: ")
        if var == "q":
            break         
    
    

