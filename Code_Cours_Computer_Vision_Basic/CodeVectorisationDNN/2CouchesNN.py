# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:50:11 2022

@author: shouhou
"""


from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


#Ecrire une fonction d'initialisation
#n0 le nombre d'entrée du résea
#n1 nombre de neurones dans la couche 1
#n2 : nombre de neurones dans la couches 2 la sortie'
def initialisation(n0, n1, n2):
    W1 = np.random.randn(n1, n0) # on donne la dimension de x pour que le vecteur w prends le meme nombre de variable
    b1= np.random.randn(n1, 1) # ici on rentre un nombre réel
    W2 = np.random.randn(n2, n1) # on donne la dimension de x pour que le vecteur w prends le meme nombre de variable
    b2= np.random.randn(n2, 1) # ici on rentre un nombre réel
    
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2,
        }
    return parameters






def model_forward_propagation(X,parametres):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1': A1,
        'A2': A2
    }

    return activations



# fonction cout
def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A+epsilon) - (1 - y) * np.log(1 - A + epsilon))




# fonction de gradients

def gradients_back_propagation(X, y, parametres, activations):

    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    
    return gradients


def update(gradients, parametres, learning_rate):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres


def predict(X, parametres):
  activations = model_forward_propagation(X, parametres)
  A2 = activations['A2']
  return A2 >= 0.5



def neural_network(X_train, y_train, n1, learning_rate = 0.1, n_iter = 1000):

    # initialisation parametres
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    np.random.seed(0)
    parametres = initialisation(n0, n1, n2)

    train_loss = []
    train_acc = []
    history = []

    # gradient descent
    for i in tqdm(range(n_iter)):
        
        activations = model_forward_propagation(X_train, parametres)
        A2 = activations['A2']

        # Plot courbe d'apprentissage
        train_loss.append(log_loss(y_train.flatten(), A2.flatten()))
        y_pred = predict(X_train, parametres)
        train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
        
        history.append([parametres.copy(), train_loss, train_acc, i])

        # mise a jour
        gradients = gradients_back_propagation(X_train, y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()

    return parametres


from utilities import *

X_train, y_train, X_test, y_test = load_data()

y_train = y_train.T

y_test = y_test.T

print(X_train.shape) # 1000 PHOTO
print(np.unique(y_train, return_counts=True)) # data set equilibré 500 PHOTO DE CHAT ET 500 PHOTO DE CHIEN


print(X_test.shape) # 200 PHOTO
print(np.unique(y_test, return_counts=True)) # data set equilibré 100 PHOTO DE CHAT ET 100 PHOTO DE CHIEN



X_train = X_train.T
X_train_reshape=X_train.reshape(-1, X_train.shape[-1] ) / X_train.max()
print(X_train_reshape.shape)
    
#X_train_reshape=X_train.reshape(X_train.shape[0],-1)

X_test=X_test.T
X_test_reshape=X_test.reshape(-1, X_test.shape[-1]) / X_train.max()
print(X_test_reshape.shape)

print(y_train.shape) # 1000 label
print(y_test.shape) # 200 PHOTO

# il faut toujours normaliser les donners quand on utilise le diagramme de desciente
# on utilise le min max / 255


#W,b= neural_network(X_train_reshape, y_train,n1=2, learning_rate=0.01, n_iter= 1000)








def artificial_neuron2(X_train, y_train, X_test, y_test, n1, learning_rate = 0.1, n_iter = 1000):
    # initialisation W, b
    # initialisation parametres train
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    np.random.seed(0)
    parametres1 = initialisation(n0, n1, n2)


 # initialisation parametres test
    n0 = X_test.shape[0]
    n2 = y_test.shape[0]
    np.random.seed(0)
    parametres2 = initialisation(n0, n1, n2)
 
 
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    history = []

    for i in tqdm(range(n_iter)):
        
        activations1 = model_forward_propagation(X_train, parametres1)
        # mise a jour
        gradients1 = gradients_back_propagation(X_train, y_train, parametres1, activations1)
        parametres1 = update(gradients1, parametres1, learning_rate)
     
        
        activations2 = model_forward_propagation(X_test, parametres2)
        # mise a jour
        gradients2 = gradients_back_propagation(X_test, y_test, parametres2, activations2)
        parametres2 = update(gradients2, parametres2, learning_rate)

        if i %10 == 0:
            # Train
            # Plot courbe d'apprentissage
            train_loss.append(log_loss(y_train.flatten(), activations1['A2']))
            y_pred = predict(X_train, parametres1)
            train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
            history.append([parametres1.copy(), train_loss, train_acc, i])


            # Test
            test_loss.append(log_loss(y_test.flatten(), activations2['A2']))
            y_pred = predict(X_test, parametres2)
            test_acc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))

     


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.legend()
    plt.show()

    return (parametres1, parametres2)


artificial_neuron2(X_train_reshape, y_train, X_test_reshape, y_test,n1=32, learning_rate = 0.01, n_iter=10000)
