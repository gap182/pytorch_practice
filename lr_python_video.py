import numpy as np 
import matplotlib.pyplot as plt 



#step1 

def initialize_parameters(lenw):
    w = np.random.randn(1,lenw)
    b = 0
    return w, b

#step2

def forward_prop(X,w,b):
    z = np.dot(w,X)+b
    return z

#step3

def cost_fun(z,y):
    m = y.shape[1]
    cost = (1/(2*m))*np.sum(np.square(z-y))
    return cost

#step4

def backprop(X,y,z):
    m = y.shape[1]
    dz = (1/m)*(z-y)
    dw = dz@X.T
    db = np.sum(dz)
    return dw, db

#step5

def grad_des(w,b,dw,db,lr):
    w -= lr*dw
    b -= lr*db
    return w,b

#step6

def linear_regression_model(X_train, y_train, lr, epochs):
    lenw = X_train.shape[0]
    w,b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    # m_val = y_val.shape[1]
    for i in range(1,epochs+1):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_fun(z_train, y_train)
        dw, db = backprop(X_train, y_train, z_train)
        w,b = grad_des(w,b,dw,db,lr)

        if i%50==0:
            costs_train.append(cost_train)
            print(f'Loss for epoch {i} is: {cost_train}')
        
        MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train))

        # z_val = forward_prop(X_val, w, b)
        # cost_val = cost_fun(z_val, y_val)
        # MAE_val = (1/m_val)*np.sum(np.abs(z_train-y_train))

lr = 0.001
epochs = 400


np.random.seed(10)

x1 = 10*np.random.rand(100).reshape(1,-1)
x2 = 10*np.random.rand(100).reshape(1,-1)
x3 = 10*np.random.rand(100).reshape(1,-1)
x4 = 10*np.random.rand(100).reshape(1,-1)

e = np.random.randn(100)

param = [2.0,3.0,-4.0,7.5]

y = (x1*param[0]+x2*param[1]+x3*param[2]+x4*param[3] + e)
# param_w = np.random.randn(4,1)

data = np.concatenate((x1,x2,x3,x4))

linear_regression_model(data, y, lr, epochs)


