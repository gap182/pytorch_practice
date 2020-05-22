## Aquí implementaré regresión lineal con pytorch pero de manera manual

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch

# print(torch.__version__)
# usar gpu o cpu de acuerdo si hay o no una tarjeta que soporte cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(device)

torch.manual_seed(10)

## definición de datos que tengan una característica lineal

x1 = 10*np.random.rand(100).reshape(1,-1)
x2 = 10*np.random.rand(100).reshape(1,-1)
x3 = 10*np.random.rand(100).reshape(1,-1)
x4 = 10*np.random.rand(100).reshape(1,-1)

e = np.random.randn(100).reshape(1,-1)

param = [2.0,3.0,-4.0,7.5,0.0]

y = (x1*param[0]+x2*param[1]+x3*param[2]+x4*param[3] + param[4] + e)

data = np.concatenate((x1,x2,x3,x4))

data = torch.from_numpy(data)
y = torch.from_numpy(y)
w = torch.randn((1,4))
b = torch.randn()

# print(data.shape, y.shape)

#definir funciones del modelo

#forward

def forward(X,w,b):
    #las dimensiones de w serían (1,pará)
    #las dimensiones de X serían (pará, #datos)
    #las dimensiones de z serían (1, #datos)
    z = torch.mm(w,X) + b
