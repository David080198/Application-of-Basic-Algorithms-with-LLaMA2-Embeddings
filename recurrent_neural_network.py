# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r7ETXPlZPjG-GqvWFkSngAKNnaLparR-
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# batch_size x sequence_length x input_size
# Imports
import torch
import time
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!

df = pd.read_csv("llama2_embeddings.csv")
df

df_embeddings = df.iloc[:,:-1]
etiquetas = df.iloc[:,-1]
etiquetas_ls = etiquetas.tolist()
#print(etiquetas_ls)
nuevas_etiquetas = []
for i in etiquetas_ls:
  if i == "[1, 0, 0, 0, 0]":
    nuevas_etiquetas.append(0)
  elif  i == "[0, 1, 0, 0, 0]":
    nuevas_etiquetas.append(1)
  elif  i == "[0, 0, 1, 0, 0]":
    nuevas_etiquetas.append(2)
  elif  i == "[0, 0, 0, 1, 0]":
    nuevas_etiquetas.append(3)
  elif  i == "[0, 0, 0, 0, 1]":
    nuevas_etiquetas.append(4)

#print(nuevas_etiquetas)

nuevo_dataset = df_embeddings
nuevo_dataset['etiquetas'] = nuevas_etiquetas
nuevo_dataset

# Dividir el DataFrame en características (X) y variable objetivo (y)
X = nuevo_dataset.iloc[:,:-1]
#print(X)
y = nuevo_dataset.iloc[:,-1]
#print(y)

tamano = X.iloc[0]
tamano1 = tamano.tolist()
tamano1 = len(tamano1)

# Hyperparameters
input_size = tamano1 # Representa la dimension de los datos de entrada en cada paso de timepo. En este caso, se espera que los datos de entrada tengan una dimension de 28
hidden_size = 128 # Especifica el numero de unidades en la capa oculta de la RNN, en este caso hay 256 unidades en la capa oculta.
num_layers = 4 # Indica la cantidad de capas en la RNN. Aquí, se han especificado dos capas de RNN
num_classes = 5 # Representa el número de clases en el problema de clasificación al que se aplica la RNN
learning_rate = 0.0001
batch_size = 64
num_epochs = 1

class RNNClasificacion(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClasificacion, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        # Inicializar hidden state y cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Propagar la entrada a través de la RNN
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)

        # Obtener la salida del último paso de tiempo
        out = self.fc(out[:, -1, :])
        return out

def crear_directorio(nombre_carpeta):
    directorio_actual = os.getcwd()
    print("El directorio actual es:", directorio_actual)
    ruta_nueva_carpeta = os.path.join(directorio_actual, nombre_carpeta)
    # Verificar si la carpeta ya existe
    if not os.path.exists(ruta_nueva_carpeta):
        # Crear la carpeta si no existe
        os.mkdir(ruta_nueva_carpeta)
        print("Se creó la carpeta", nombre_carpeta, "en", directorio_actual)
    else:
        print("La carpeta", nombre_carpeta, "ya existe en", directorio_actual)

    ruta_modificada = ruta_nueva_carpeta.replace("\\","/")
    return ruta_modificada

def ejecucion_modelo(X, y, input_size, hidden_size, num_layers, num_classes,learning_rate,batch_size,num_epochs,porcentage):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=porcentage, random_state=42)
  # Convertir los conjuntos de datos en tensores de PyTorch
  X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
  X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

  X_train_tensor = X_train_tensor.unsqueeze(1)
  X_test_tensor = X_test_tensor.unsqueeze(1)

  # Crear conjuntos de datos de PyTorch
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

  # Crear DataLoaders para conjuntos de entrenamiento y prueba
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  # Instanciar el modelo
  modelo = RNNClasificacion(input_size, hidden_size, num_layers, num_classes).to(device)

  # Definir función de pérdida y optimizador
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)
  losses = []
  # Iterar sobre los datos de entrenamiento
  print(f"Entrenando modelo con {num_epochs}, tasa de aprendizaje {learning_rate} y porcentaje del corpus de {1-porcentage}")
  for epoch in tqdm(range(num_epochs),desc="Epocas "):
      modelo.train()  # Establecer el modelo en modo de entrenamiento
      epoch_loss = 0
      for inputs, labels in train_loader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          inputs = inputs.float()  # Convertir a tipo float si es necesario
          labels = labels.long()  # Convertir a tipo long (para índices) si es necesario

          # Forward pass
          outputs = modelo(inputs)
          loss = criterion(outputs, labels)
          epoch_loss += loss.item()

          # Backward pass y optimización
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      epoch_loss /= len(train_loader)
      losses.append(epoch_loss)
      # Calcular la precisión en el conjunto de entrenamiento
      with torch.no_grad():
          modelo.eval()  # Establecer el modelo en modo de evaluación
          correct = 0
          total = 0
          for inputs, labels in train_loader:
              inputs = inputs.to(device)
              labels = labels.to(device)
              inputs = inputs.float()
              labels = labels.long()
              outputs = modelo(inputs)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
          train_accuracy = correct / total

      all_predictions = []
      all_labels = []
      # Calcular la precisión en el conjunto de prueba
      with torch.no_grad():
          correct = 0
          total = 0
          for inputs, labels in test_loader:
              inputs = inputs.to(device)
              labels = labels.to(device)
              inputs = inputs.float()
              labels = labels.long()
              outputs = modelo(inputs)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

              # Agregar las predicciones y las etiquetas reales a las listas
              all_predictions.extend(predicted.tolist())
              all_labels.extend(labels.tolist())
          test_accuracy = correct / total

  ##############################################

  timestamp = time.strftime("%Y%m%d_%H%M%S")
  ruta_figura_incom = crear_directorio("Recurrent_Neural_Network")

  nombre_carpeta = f"RNN_{num_epochs}_{learning_rate}_{porcentage}"
  os.makedirs(os.path.join(ruta_figura_incom, nombre_carpeta), exist_ok=True)
  ruta_figura_incom = ruta_figura_incom + "/" + nombre_carpeta

  ruta_figura = f"{ruta_figura_incom}/matriz_confusion_{num_epochs}_{learning_rate}_{porcentage}.png"
  ruta_guardar_modelo = f"{ruta_figura_incom}/modelo_entrenado_{num_epochs}_{learning_rate}_{porcentage}.pth"
  ruta_metricas = f"{ruta_figura_incom}/metricas_{num_epochs}_{learning_rate}_{porcentage}.json"
  ruta_loss = f"{ruta_figura_incom}/loss_function_{num_epochs}_{learning_rate}_{porcentage}.png"

  ############################################

  conf_matrix = confusion_matrix(all_labels, all_predictions)
  plt.figure(figsize=(8, 6))
  sns.heatmap(conf_matrix, annot=True,fmt='d', cmap='Blues')
  # Añadir etiquetas y título
  plt.xlabel('Predicted labels')
  plt.ylabel('True labels')
  plt.title('Confusion Matrix')
  plt.savefig(ruta_figura)
  # Mostrar el gráfico
  #plt.show()

  metricas = {}
  # Calcular métricas de rendimiento
  accuracy = accuracy_score(all_labels, all_predictions)
  precision = precision_score(all_labels, all_predictions, average='macro')
  recall = recall_score(all_labels, all_predictions, average='macro')
  f1 = f1_score(all_labels, all_predictions, average='macro')


  metricas['accuracy'] = accuracy
  metricas['precission'] = precision
  metricas['recall'] = recall
  metricas['f1'] = f1
  metricas['loss'] = loss.item()
  metricas['train_accunrancy'] = train_accuracy
  metricas['test_accyrancy'] = test_accuracy
  with open(ruta_metricas, 'w') as archivo_json:
    json.dump(metricas, archivo_json)

  # Imprimir resultados
  print("Matriz de Confusión:")
  print(conf_matrix)
  print("Accuracy:", accuracy)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1-score:", f1)

  # Imprimir métricas de rendimiento
  print(f'Época {epoch+1}/{num_epochs}, Pérdida: {loss.item():.4f}, Precisión (Entrenamiento): {train_accuracy:.4f}, Precisión (Prueba): {test_accuracy:.4f}')

  # Graficar la pérdida en función de las épocas
  print(losses)
  print(len(losses))
  print(len(range(1,num_epochs+1)))
  plt.plot(range(1, num_epochs+1), losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss over Epochs')
  plt.legend()
  plt.savefig(ruta_loss)
  plt.show()

  torch.save(modelo.state_dict(), ruta_guardar_modelo)

# Hyperparameters
input_size = tamano1 # Representa la dimension de los datos de entrada en cada paso de timepo. En este caso, se espera que los datos de entrada tengan una dimension de 28
hidden_size = 128 # Especifica el numero de unidades en la capa oculta de la RNN, en este caso hay 256 unidades en la capa oculta.
num_layers = 5 # Indica la cantidad de capas en la RNN. Aquí, se han especificado dos capas de RNN
num_classes = 5 # Representa el número de clases en el problema de clasificación al que se aplica la RNN
learning_rate = 0.0001
batch_size = 64
num_epochs = 1
#ejecucion_modelo(X, y, input_size, hidden_size, num_layers, num_classes,learning_rate,batch_size,num_epochs)

epocas = [100,300,500,700,1000]
learning_rate = [0.0001,0.00001]
test_porcentage = [0.3,0.2,0.1]

for test in test_porcentage:
	for epoca in epocas:
  		for learning in learning_rate:
    			ejecucion_modelo(X, y, input_size, hidden_size, num_layers, num_classes,learning,batch_size,epoca,test)

