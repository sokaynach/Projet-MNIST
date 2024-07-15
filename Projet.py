#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:56:15 2023

@author: sokaynachaoui
"""

#Importation des librairies
##Partie 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from IPython import display
import time
import plotly.express as px
##Partie 2
import tensorflow as tf
from tensorflow.keras import layers, models
##Partie 3
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#### Partie 1 : Préparation des jeu de données

#Affichage de toutes les colonnes 
pd.set_option("display.max_columns", None)

#Chargement des csv d'entrainement et de test
mnist = pd.read_csv("/Users/sokaynachaoui/Documents/projet d'été/train.csv",delimiter=",")
test_data = pd.read_csv("/Users/sokaynachaoui/Documents/projet d'été/test.csv",delimiter=",").values
data = pd.read_csv("/Users/sokaynachaoui/Documents/projet d'été/train.csv")

#Affichage du nombre de ligne et colonnes
print(mnist.shape)
print(test_data.shape)

# Affichage des images du jeu d'entrainement
with open("/Users/sokaynachaoui/Documents/projet d'été/train.csv", 'r') as f:
    reader = csv.reader(f, delimiter = ",")
    next(reader) # Passe la première ligne
    ligne = f.readlines() # Extrait les lignes du csv
    for l in ligne[1:][500:503]: # Extraction des données à partir de la deuxième colonne et de la ligne 501 à la ligne 504 (choisis au hasard pour vérification)
        split_lines = [int(x) for x in l.split(",")] # Retire les ' pour reutiliser la ligne comme une matrice
        newline = split_lines[1:] # Suppression du premier élément qui ne correspond pas a un pixel de l'image mais au chiffre représenté sur l'image
        #Verification
        print(newline)
        #Création d'une matrice 28x28 avec chaque ligne
        matrice = np.reshape(newline,(28,28))
        print(matrice) #Verification
        #Affichage de l'image
        #Mise en place du graphe 
        plt.figure(figsize=(5,5))
        plt.imshow(matrice)
        plt.title(split_lines[0]) # Chiffre corespondant en titre
        display.clear_output(wait=True)
        pause_time = 0.2
        time.sleep(pause_time)
 
#Préparation des données pour CNN
# Creation de deux nouvelles data frame repertoriant chaque images et chaque labels
train_data, train_label = mnist.iloc[:, 1:].values, mnist['label'].values
print(train_data.shape)
print(train_label.shape)

train_data=train_data.reshape(-1,28,28,1)

test_data=test_data.reshape(-1,28,28,1)

# Normaliser les valeurs des pixels entre 0 et 1
train_data, test_data = train_data / 255.0, test_data / 255.0

#Préparation des données pour SVM

# X matrice 28x28 et y label
X = data.drop('label', axis=1) 
y = data['label']

X=X.values.reshape(-1,28,28,1)
X = X / 255.0

##  Visualisation (Camembert de répartition des labels) 

# Convertir train_label en une série Pandas
train_label_series = pd.Series(train_label)

# Obtenir les comptages des valeurs
label_dis = train_label_series.value_counts()

# Extrait le nombre d'années de chaque experience de experience_dis 
index = label_dis.index

# Creation du diagramme en camembert
fig = px.pie(values=label_dis, names=index, color=index)

# Création du titre
fig.update_layout(title_text='Distribution des labels' )

# Affichage du diagramme sur un navigateur web
fig.show(renderer='browser')

#### Partie 2 : Mise en place du CNN

## Création du modèle CNN
# Création du modèle séquentiel
model = models.Sequential()
# Création d'une première couche de convolution avec 32 filtres de taille (3, 3), 
#une fonction d'activation ReLU, et une forme d'entrée de (28, 28, 1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Reduction de la dimension spatiale avec couche de max-pooling avec fenêtre (2,2)
model.add(layers.MaxPooling2D((2, 2)))
# Autre couche de convolution mais 64 filtres
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Autre Max-pooling
model.add(layers.MaxPooling2D((2, 2)))
# couche de convolution 64 filtres
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

## Ajout des couches de classification
#Couche d'applatissement de 2D en, 1D (Couche completement connecté)
model.add(layers.Flatten())
#Couche dense avec 64 filtres
model.add(layers.Dense(64, activation='relu'))
#Couche dense de sortie avec 10 neurones pour les dix classes
model.add(layers.Dense(10))  # 10 classes de chiffres

## Compiler le modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## Entraîner le modèle
model.fit(train_data, train_label, epochs=10) #10 epochs d'entrainement

## Faire des prédictions sur les données de test
predictions = model.predict(test_data)

## Retrouver la classe prédite

softmax_predictions = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))

softmax_predictions /= np.sum(softmax_predictions, axis=1, keepdims=True)

predicted_classes = np.argmax(softmax_predictions, axis=1)

print(predicted_classes)

# Verification des prédictions d'image test (21000 n'est qu'un exemple)
plt.figure(figsize=(5,5))
plt.imshow(test_data[21000])
plt.title(predicted_classes[21000]) # Chiffre corespondant en titre
display.clear_output(wait=True)
pause_time = 0.2
time.sleep(pause_time)



### Partie 3 : Mise en place du SVM

##Couper le fichier en 80% et 20% pour l'entrainement et le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Appliquez une transformation de déroulement (flatten) pour chaque image
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


svm_model = SVC(kernel='rbf', gamma='scale')

##Entrainement du modèle
start_time = time.time() 
svm_model.fit(X_train, y_train)
training_time = time.time() - start_time #Temps d'entrainement du modèle

##Prédiction du modèle
start_time = time.time()
y_pred = svm_model.predict(X_test)
prediction_time = time.time() - start_time #Temps de prédiction du modèle

##Calcul de la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitude du modèle SVM : {accuracy}')

# Affichez les temps de traitement
print(f'Temps d\'entraînement : {training_time} secondes')
print(f'Temps de prédiction : {prediction_time} secondes')

# Verification des prédictions d'image test
# Utilisation de reshape pour redonner sa forme d'origine à X_test
X_test = X_test.reshape(X_test.shape[0], 28, 28)
plt.figure(figsize=(5,5))
plt.imshow(X_test[2806]) #Pris au hasard pour vérification
plt.title(y_pred[2806]) # Chiffre corespondant en titre
display.clear_output(wait=True)
pause_time = 0.2
time.sleep(pause_time)
