# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:19:58 2023

@author: eliza
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Datos de ejemplo
df = pd.read_csv('new_data_01.csv')
le = LabelEncoder()
df['Posiciones'] = le.fit_transform(df['Posiciones']) 
y = df['Posiciones'].to_numpy()  # O df['columna_ejemplo'].values

#msno.matrix(df)
#plt.show()
X = df[['brand','brandId','cant_img','cat_atrib','catalog_product_id',
'Cat_Price','adId','base_price','date_created','deal_ids','direction',
'has_video','health','id','integrator',
'is_financeable','kilometers','kilometraje','listing_type_id','max_size',
'model','modelId','motor','nickname','original_price','parent_item_id',
'positionBycategory','price','seller_id','seller_reputation/metrics/cancellations/period',
'seller_reputation/metrics/cancellations/value','seller_reputation/metrics/claims/period',
'seller_reputation/metrics/claims/rate','seller_reputation/metrics/claims/value',
'seller_reputation/metrics/delayed_handling_time/period',
'seller_reputation/metrics/delayed_handling_time/rate',
'seller_reputation/metrics/sales/completed',
'seller_reputation/transactions/canceled','seller_reputation/transactions/completed',
'seller_reputation/transactions/ratings/neutral','seller_reputation/transactions/ratings/positive',
'seller_reputation/transactions/total','title','positionByBrandModelYear','totalGold','totalGoldPremium',
'totalSilver','total_attributes','total_images',
'traction','version','versionId','visits','yearId','year_00'
]]
categoricas = ['brand','brandId','cant_img','cat_atrib','catalog_product_id',
'Cat_Price','adId','base_price','date_created','deal_ids','direction',
'has_video','health','id','integrator',
'is_financeable','kilometers','kilometraje','listing_type_id','max_size',
'model','modelId','motor','nickname','original_price','parent_item_id',
'positionBycategory','price','seller_id','seller_reputation/metrics/cancellations/period',
'seller_reputation/metrics/cancellations/value','seller_reputation/metrics/claims/period',
'seller_reputation/metrics/claims/rate','seller_reputation/metrics/claims/value',
'seller_reputation/metrics/delayed_handling_time/period',
'seller_reputation/metrics/delayed_handling_time/rate',
'seller_reputation/metrics/sales/completed',
'seller_reputation/transactions/canceled','seller_reputation/transactions/completed',
'seller_reputation/transactions/ratings/neutral','seller_reputation/transactions/ratings/positive',
'seller_reputation/transactions/total','title','positionByBrandModelYear','totalGold','totalGoldPremium',
'totalSilver','total_attributes','total_images',
'traction','version','versionId','visits','yearId','year_00'
] # Reemplaza con los nombres de tus características categóricas
for feature in categoricas:     
    X[feature] = le.fit_transform(X[feature]) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Crear un modelo de CNN
model = keras.Sequential([
    layers.Reshape(target_shape=(55, 1), input_shape=(55,)),  # Añadir una dimensión para los canales
    layers.Conv1D(32, 3, activation='relu', input_shape=(67, 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')  #  salida
])

# # Compilar el modelo
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# # Entrenar el modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.3)
y_pred = model.predict(X_test)
# Calcular la precisión del modelo
# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
# Convierte las probabilidades a etiquetas de clase
y_pred_labels = np.argmax(y_pred, axis=1)
# Calcula la precisión
accuracy = accuracy_score(y_test, y_pred_labels)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
# Calculate recall
recall = recall_score(y_test, y_pred_labels, average=None)
print("Recall:", recall)
# Calculate F1 score
f1 = f1_score(y_test, y_pred_labels, average=None)
print("F1 Score:", f1)

confusion = confusion_matrix(y_test, y_pred_labels)  
# # Etiquetas para los ejes
labels = df['Posiciones'].unique()
# # Crear una figura
plt.figure(figsize=(8, 6))
# # Crear el mapa de calor de la matriz de confusión
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
# # Configuración de etiquetas y título
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
# # Mostrar la figura
plt.show()

# saving and loading the .h5 model
 
# save model
model.save('gfgModel.h5')
print('Model Saved!')
 
# load model
savedModel=load_model('gfgModel.h5')
savedModel.summary()