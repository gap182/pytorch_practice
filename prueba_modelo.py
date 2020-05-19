import numpy as np 
import matplotlib.pyplot as plt

lr = 0.001
epochs = 500

x = np.linspace(0,10,100)
y = np.linspace(0,10,100)
e = np.random.rand(100)

y += e

param_w = np.random.randn()
param_b = np.random.randn()

pred = x*param_w + param_b


loss = ((pred-y)**2).sum()/len(x)

print('Initial loss: {}'.format(loss))

for epoch in range(epochs):

    dl_dw = (2*(pred-y)*x).sum()/len(x)
    dl_db = (2*(pred-y)).sum()

    param_w = param_w - lr*dl_dw
    param_b = param_b - lr*dl_db

    pred = x*param_w + param_b 

    loss = ((pred-y)**2).sum()/len(x)

    if epoch%50 == 0:
        print('The Loss in the epoch: {} is: {}'.format(epoch, loss))


plt.scatter(x,y,color='r',s=0.5)
plt.plot(x,pred,color='b')
plt.show()