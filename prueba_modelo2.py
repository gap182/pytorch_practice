import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

np.random.seed(10)

lr = 0.001
epochs = 400

x1 = 10*np.random.rand(100).reshape(1,-1)
x2 = 10*np.random.rand(100).reshape(1,-1)
x3 = 10*np.random.rand(100).reshape(1,-1)
x4 = 10*np.random.rand(100).reshape(1,-1)

e = np.random.randn(100)

param = [2.0,3.0,-4.0,7.5]

y = (x1*param[0]+x2*param[1]+x3*param[2]+x4*param[3] + e).reshape(100,1)

## Mostrar gráficas
# fig, ax = plt.subplots(2,2)
# ax[0,0].scatter(x1,y)
# ax[0,0].set_title('x1')
# ax[0,1].scatter(x2,y)
# ax[0,1].set_title('x2')
# ax[1,0].scatter(x3,y)
# ax[1,0].set_title('x3')
# ax[1,1].scatter(x4,y)
# ax[1,1].set_title('x4')
# fig.tight_layout()

# plt.show()

param_w = np.random.randn(4,1)

data = np.concatenate((x1,x2,x3,x4)).T

pred = data@param_w

loss = ((pred-y)**2).sum()/100
print(f'Loss inicial: {loss}')

# print(pred.shape)

dl_dw = 2*data.T@(pred-y)/100

# dl_dw = np.zeros((4,1))

# for i in range(4):
#     for j in range(100):
#         dl_dw[i] += (pred[j][0]-y[j][0])*data[j,i]


for i in range(epochs):
    dl_dw = 2*data.T@(pred-y)/100
    # print(param_w.shape, dl_dw.shape)
    param_w -= lr*dl_dw
    # print(data.shape, param_w.shape)
    pred = data@param_w

    # print(pred.shape, y.shape, (pred-y).shape)
    loss = ((pred-y)**2).sum()/100

    if i%50 == 0:
        print(f'El error para la época {i} es: {loss}')