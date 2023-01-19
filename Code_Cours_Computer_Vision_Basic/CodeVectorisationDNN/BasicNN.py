# Dans ce code nous avons developper neuronne artificielle ave regression logistique c'est un model simple car il traite que les modele lineare

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score






#Ecrire une fonction d'initialisation
def initialisation(X):
    W = np.random.randn(X.shape[1], 1) # on donne la dimension de x pour que le vecteur w prends le meme nombre de variable
    b= np.random.randn(1) # ici on rentre un nombre réel
    return(W, b)






def model(X,W,b):
    Z = X.dot(W)+ b
    A = 1 / (1+np.exp(-Z))
    return A



# fonction cout
def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A+epsilon) - (1 - y) * np.log(1 - A + epsilon))




# fonction de gradients

def gradients(A,X, y):
    dW= 1 / len(y) * np.dot(X.T, A - y)
    db= 1 / len(y) * np.sum(A - y)
    return (dW, db)






# implementer la fonction update

def update(dW, db, W, b, Learning_rate):
    W = W - Learning_rate * dW
    b = b - Learning_rate * db
    return (W, b)

# definir la fonction de prediction

def predict(X,W,b):
    A=model(X, W, b)
    # imprimer la probabilité
   # print(A)
    return A >= 0.5

from tqdm import tqdm

def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    W,b =initialisation(X)
    
    Loss =[] # cree une liste pour visualiser levauation cout
    acc = []
    #print(W.shape)
    #print(b.shape)
    for i in tqdm(range(n_iter)):
       # activation
        A = model(X,W,b)
      #  A.shape
      #Calcul le cout
      
        Loss.append(log_loss(A, y))
        
        # calcul occuracy
        y_pred = predict(X, W, b)
        acc.append(accuracy_score(y, y_pred))

       # print(f'la performence de lalgorithme est {accuracy_score(y, y_pred)}')
       #Mise a jour
        dW, db = gradients(A, X, y)
       # print(dW.shape)
        #print(db.shape)
        W, b= update(dW, db, W,b, learning_rate)
    
    # calculer la performence de la'lgorithme
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(Loss)
    plt.subplot(1,2,2)

    plt.plot(acc)
    plt.show()
    return (W,b)
    
# W,b= artificial_neuron(X, y)

# print(W,b)

#  #utiliser ces parametre sur une nouvelle plante
# new_plant = np.array([2, 1])


# #tracer les frontiére de decision
# x0=np.linspace(-1, 4, 100)
# x1= (-W[0] * x0 - b) / W[1]


# plt.scatter(X[:,0],X[:,1], c=y, cmap='summer')
# plt.scatter(new_plant[0],new_plant[1], c='red')
# plt.plot(x0, x1, c='orange', lw=3)
# plt.show()
# predict(new_plant, W,b)


from utilities import *

X_train, y_train, X_test, y_test = load_data()
print(X_train.shape) # 1000 PHOTO
print(y_train.shape) # 1000 label
print(np.unique(y_train, return_counts=True)) # data set equilibré 500 PHOTO DE CHAT ET 500 PHOTO DE CHIEN


print(X_test.shape) # 200 PHOTO
print(y_test.shape) # 200 PHOTO
print(np.unique(y_test, return_counts=True)) # data set equilibré 100 PHOTO DE CHAT ET 100 PHOTO DE CHIEN


# Afficher les 10 premiere photo de cette dataset

plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()


# TO DO 
# 1. Normaliser le train_set et le test_set (0-255 -> 0-1)
# 2. flatten() les variables du train_set et du test_set (64x64 -> 4096)
# 3. Entrainer le modele sur le train_set (tracer la courbe d'apprentissage, trouver les bons hyper-params)
# (si vous rencontrez un probleme avec le log_loss, utiliser la fonction de sklearn a la place !)
# 4. Évaluer le modele sur le test_set (tracer également la courbe de Loss pour le test_set)
# 5. Partager vos conclusions dans les commentaires !

#entrainement de model
# W,b= artificial_neuron(X_train, y_train) cava pas marcher car on a 1000 65 65
#" IL FAUT RESHAPE XTRAIN Y TRAIN" donc on utilise l'un des formulle suivante: 
    
X_train_reshape=X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) / X_train.max()

X_train_reshape.shape
    
#X_train_reshape=X_train.reshape(X_train.shape[0],-1)


X_test_reshape=X_test.reshape(X_test.shape[0],-1) / X_train.max()

X_test_reshape

# il faut toujours normaliser les donners quand on utilise le diagramme de desciente
# on utilise le min max / 255


W,b= artificial_neuron(X_train_reshape, y_train, learning_rate=0.01, n_iter= 1000)

# pour qu'on puisse savoir si notre model est en overfiting il faut qu'on compare avec le test images, c'est pour cela nous modifions la fonction comme suit




def artificial_neuron2(X_train, y_train, X_test, y_test, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):
        A = model(X_train, W, b)

        if i %10 == 0:
            # Train
            train_loss.append(log_loss(A, y_train))
            y_pred = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))

            # Test
            A_test = model(X_test, W, b)
            test_loss.append(log_loss(A_test, y_test))
            y_pred = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred))

        # mise a jour
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)


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

    return (W, b)


W, b = artificial_neuron2(X_train_reshape, y_train, X_test_reshape, y_test, learning_rate = 0.01, n_iter=10000)


# le model trop faible pour obtenir de bon resultat