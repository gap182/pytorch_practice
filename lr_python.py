# Este sería la regresión lineal solo con python

import numpy as np 
import pandas as pd 


## Importando los datos Boston Housing

df = pd.read_csv('HousingData.csv')

## llenando los valores nulos con el promedio de cada columna

for i in df.columns:
    df[i].fillna(value=df[i].mean(), inplace=True)

## Verificar que no hay columnas con datos vacíos

# print(df.isnull().sum())

## guardando los datos en un diccionario con los datos tipo array
## también en una lista para tenerlos en 
variables = {}
data = []
for name in df.columns:
    variables[name] = np.array(df[name])
    data.append(df[name])

#Será mejor almacenar los datos como una matriz

var_dep = df.drop(['MEDV'], axis=1)
var_dep = np.array(var_dep)
y = df['MEDV']
y = np.array(y).reshape(len(df),1)

# data2 = np.array(var_dep)

#El modelo sería algo así y_i = w1*x1_i+w2*x2_i+...+b

#Inicialicemos los parámetros de manera aleatoria

param_w = np.random.randn(len(df.columns)-1,1)
param_b = np.random.randn(1)

pred = var_dep@param_w + param_b

loss = ((pred-y)**2).sum()/len(df)

## Cálculo de los gradientes



# loss = (pred-y)

# print(y.shape, pred.shape)


# print(pred)

# print(param)ta2@param

# loss = ()

# print(pred)

## calculando la predición

# print(data2.shape,param.shape)
# print(param)