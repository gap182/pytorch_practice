# Este sería la regresión lineal solo con python

import numpy as np 
import pandas as pd 


## Importando los datos Boston Housing
lr = 0.01
epochs = 500
df = pd.read_csv('HousingData.csv')
np.random.seed(10)
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

var_dep = df[['RM', 'LSTAT']]
var_dep = np.array(var_dep)

y = df['MEDV']
y = np.array(y).reshape(len(df),1)

# # data2 = np.array(var_dep)

# #El modelo sería algo así y_i = w1*x1_i+w2*x2_i+...+b

# #Inicialicemos los parámetros de manera aleatoria

param_w = np.random.randn(2,1)
param_b = np.random.randn()

# print(var_dep.shape, param_w.shape)

# print(var_dep[0,:])
# print(df.head())
# print(param_w)
# print(param_b)

pred = var_dep@param_w + param_b

# prueba=0
# for i in range(13):
#     prueba = prueba + var_dep[0,i]*param_w[i]
#     # print(var_dep[0,i], param_w[i])
# prueba = prueba+param_b

# print(pred[0],prueba)
# loss2 = 0.0
# for i in range(len(df)):
#     loss2 = loss2 + (pred[i]-y[i])**2
# loss2 = loss2/len(df)



loss = ((pred-y)**2).sum()/len(df)
print(f'Loss inicial: {loss}')

# ## Cálculo de los gradientes

# # print(var_dep[:,0].shape, pred.shape)

# # tmp = (2*var_dep[:,0].reshape(506,1)*(pred-y)).sum()
# # dl_dw = var_dep.T@(2*(pred-y))

# for epoch in range(20):

dl_dw = 2*var_dep.T@np.absolute(pred-y)/len(df)
dl_db = (2*(pred-y)).sum()/len(df)

# print(var_dep[0,:])
# print(var_dep.T[:,0])

# tmp = 0.0
# for i in range(len(df)):
#     tmp = tmp + 2*var_dep[i,0]*np.absolute(pred[i]-y[i])
#     # print(var_dep[0,i], pred[i], y[i])
# tmp = tmp/len(df)

# print(var_dep.T[:4,:4], pred[:4], y[:4])

# print(var_dep.T[0,:] == var_dep[:,0])
# print(pred)

# print(dl_dw[0], tmp)

#     # print(dl_dw.shape, dl_db.shape)

param_w = param_w - lr*dl_dw
param_b = param_b - lr*dl_db

#     # print(param_w.shape, param_b.shape)

pred = var_dep@param_w + param_b

# #     # print(pred.shape)



loss = ((pred-y)**2).sum()/len(df)
print(f'Loss después de un paso: {loss}')

# print(tmp, dl_dw[0])
# print(dl_dw)

# lr=0.001
# param_w -= lr*dl_dw
# param_b -= lr*dl_db

# pred = var_dep@param_w + param_b

# loss2 = ((pred-y)**2).sum()/len(df)

# print(loss, loss2)
# print(dl_dw.shape)



# loss = (pred-y)

# print(y.shape, pred.shape)


# print(pred)

# print(param)ta2@param

# loss = ()

# print(pred)

## calculando la predición

# print(data2.shape,param.shape)
# print(param)