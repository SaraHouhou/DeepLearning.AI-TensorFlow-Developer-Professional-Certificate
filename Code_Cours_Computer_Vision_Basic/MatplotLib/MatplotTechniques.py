# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:27:57 2022

@author: shouhou
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
#Visualiser un graphique avec matplot avec xt et yt doit avoir le meme dimension
# xt= np.linspace(0,2,10) # tableau de 10 point allant de 0 à 2
# yt=xt**2  # tableau de 10 point le carée de xt

# #On cree un expace figure si on veut dans le future jouer avec ces dimensions en inch

# plt.figure(figsize=None)

#plt.plot(xt,yt)
#si on veut faire des caractériqtique
#   label: nom de la courbe
#   lw: epaisseur du trait
#   ls: style de trait
#   c: couleur de trait

# plt.plot(xt,yt, c='red', lw=3, ls='--',label='quadratique')

# print ('dimension de xt : ', xt.shape)
# print ('dimension de yt : ', yt.shape)

# #On peut ajouter une autre courbe sur la courbe
# #plt.plot(xt,xt**3)

# #ON AJOUTE LABELS POUR LES AXES
# plt.plot(xt,xt**3, label='cubique')
# #ON AJOUTE UN TITRE DE FIGURE
# plt.title('Figure 1 des courbes')
# plt.xlabel('Axe X')
# plt.ylabel("Axe Y")
# #Ajouter une legende pour indiquer represente quoi mes courbe
# plt.legend()
# #oN PEUT ENREGISTRER NOTRE COURBE DANS UNE IMAGE 
# plt.savefig('figureTest.png')
# plt.show()


# #Generer une grille de graphique au sein d'une figure
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(xt,yt, c='red')
# plt.title("test courbe 1")
# plt.subplot(2,1,2)
# plt.plot(xt,yt, c='blue')
# plt.title("test courbe 2")
# plt.show()






#Visualiser un graphique avec matplot avec x et y en nuage de point

#1 génerer des données comprenant 100 ligne et 2 variables features
# X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
# print ('dimension de X : ', X.shape)
# print ('dimension de y : ', y.shape)
# y=y.reshape((y.shape[0],1))

# print ('dimension de reshaped y : ', y.shape)
# plt.scatter(X[:,0],X[:,1], c=y, cmap='summer')
# plt.show()
# #plt

# Graphic 3D for 3 classes===============================================================================

#telecharger la base de donnée

iris=load_iris()

#Data contient 150 donnée repartie en 4 variables
x=iris.data

#target contient 150 elem represente 3 class 0, 1, 2
y=iris.target
names=list(iris.target_names)

print(f'c contient {x.shape[0]} example et {x.shape[1]} variables')
print(f'il y a {np.unique(y).size} classes')
print(x)
print(y)

#visualiser cette data set
#parametre alpha pour controler la traansparence des point
#parametre s pour controler la taille des point
plt.scatter(x[:,0],x[:,1],c=y, alpha=0.5, s=100)
plt.xlabel('longuer de sépal')
plt.ylabel('largeur de sépal')
plt.show()
#probleme on dispose 4 variables et on peut que présenté deux
#Solution


#Histogramme
#visualiser comment ils sont distribuer nos quatres variables
plt.hist(x[:,0], bins=20)
plt.hist(x[:,1])
plt.hist(x[:,2])
plt.hist(x[:,3])
plt.show()

#Visualiser la distribution de donnée en 2 d
#Cmap pour les rendres plus joli
plt.hist2d(x[:,0], x[:,1], cmap='Blues')
plt.xlabel('longuer de sépal')
plt.ylabel('largeur de sépal')
#Afficher aussi à quoi correspond les couleurs
plt.colorbar()
plt.show()

# histo utiliser pour l'analyse d'une images
from scipy import misc

face=misc.face(gray=True)
#plt.imshow(face, cmap='gray')
#applaté un tableaux avec ravel
plt.hist(face.ravel(),bins=255)
plt.show()

#Graphique 3 D
from mpl_toolkits.mplot3d import Axes3D
#on utilise ax , donc on doit utiliser la programmation orienté objet
ax=plt.axes(projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2], c=y, alpha=0.5, s=100)

plt.xlabel('longuer de sépal')
plt.ylabel('largeur de sépal')
plt.show()

# Visualiser des fonctions mathématique en 3d
f = lambda x , y : np.sin(x) + np.cos(x+y)

# crée deux vecteur de x et y et une sorte de grille à partir de deux axe

X = np.linspace(0,5,100)
Y = np.linspace(0,5,100)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)
Z.shape
ax=plt.axes(projection='3d')

ax.plot_surface(X,Y,Z, cmap='plasma')
plt.show()

#Contour plot pour la visualuser en vu de dessus

plt.contour(X,Y,Z, 20)
plt.colorbar()

#Visualiser une image Imshow

plt.imshow(face)
plt.show()

# elle prends la transposer de X, ici on a afficher les 4 variable de iris
plt.imshow(np.corrcoef(x.T), cmap='Blues')
plt.colorbar()
plt.show()