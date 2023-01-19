# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:09:58 2022

@author: shouhou
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

def initialisation(dimensions):
    
    parametres = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres

def testInit(): 
    parameters = initialisation([2, 32, 32, 1])
    
    for key, val in parameters.items():
        print (key, val.shape)
        
        
def forward_propagation(X, parametres):
  
  activations = {'A0': X}

  C = len(parametres) // 2

  for c in range(1, C + 1):

    Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
    activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

  return activations
    
def testforwardprog(X): 
    parametres= initialisation([2, 32, 32, 1])
    activations = forward_propagation(X, parametres)
    for key, val in activations.items():
        print (key, val.shape)