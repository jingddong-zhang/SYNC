import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy import integrate
import pickle
from functions import *

start=timeit.default_timer()
# np.random.seed(1)

model1= ControlNet(6,24,3)
model1.load_state_dict(torch.load('ES_2.pkl'))
model2 = ControlNet(7,24,3)
model2.load_state_dict(torch.load('ES_tt.pkl'))

def init_x(t):
    return np.array([1.5-np.sin(t),-4.4-np.sin(t),0.15-np.sin(t)])
def init_y(t):
    return np.array([15+np.exp(t),-4+np.exp(t),1.5+np.exp(t)])
def g(x):
    B,m0,m1=1,-1/7,-40/7
    return m0*x+(m1-m0)*(np.abs(x+B)-np.abs(x-B))/2
def f(x):
    a,b,c,d=7,0.35,0.5,7
    x1,x2,x3 = x
    return np.array([a*(x2-x1-g(x1)),b*(x1-x2)+c*x3,-d*x2])


def run(n,dt,case,seed):
    np.random.seed(seed)
    tau = 0.1
    n0 = int(tau/dt)
    nd = 10000
    X = np.zeros([n0+n,3])
    Y = np.zeros([n0+n,3])
    U = np.zeros([n-1,3])
    for i in range(n0+1):
        t = i*dt-tau
        X[i] = init_x(t)
        Y[i] = init_y(t)
    k1,k2 = 83.675,2 #83.675
    # z = np.random.normal(0,1,n)
    z = np.random.normal(0,1,[n,3])
    for i in range(n-1):
        x = X[i+n0,:]
        x_t = X[i+n0-nd,:]
        y = Y[i+n0,:]
        y_t = Y[i+n0-nd,:]
        X[i+n0+1,:]=x+dt*f(x)
        if case == 0:
            Y[i+n0+1,:]=y+dt*f(y)+np.sqrt(dt)*z[i]*(1*np.sum(np.sin(2*(x-y)))-1*np.sum(np.sin(x_t-y_t)))
            # if i%1000==0:
            #     print('original:',np.max(np.abs(x-y)),np.max(np.abs(x_t-y_t)))
        if case == 1:
            Y[i+n0+1,:]=y+dt*(f(y)+k1*(x-y)+k2*(x_t-y_t))+\
                        np.sqrt(dt)*z[i]*(1*np.sum(np.sin(2*(x-y)))-1.0*np.sum(np.sin(x_t-y_t)))*((1+i*dt))*5
            U[i,:] = k1*(x-y)+k2*(x_t-y_t)
            # if i%100==0:
            #     print('LC:',np.max(np.abs(x-y)),np.max(np.abs(x_t-y_t)),np.max(y),np.max(k1*(x-y)+k2*(x_t-y_t)))
        if case == 2:
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((x-y,x_t-y_t))).to(torch.float32).unsqueeze(0) #np.array([i*dt]),
                u = model1(input).detach().numpy()*(x-y)
            Y[i+n0+1,:]=y+dt*(f(y)-u)+np.sqrt(dt)*z[i]*(1*np.sum(np.sin(2*(x-y)))-1*np.sum(np.sin(x_t-y_t)))
            U[i,:]= u
            if i%10000==0:
                print('NN:',np.max(np.abs(x-y)),np.max(np.abs(x_t-y_t)),np.max(u))
        if case == 3:
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((np.array([i*dt]),x-y,x_t-y_t))).to(torch.float32).unsqueeze(0) #np.array([i*dt]),
                u = model2(input).detach().numpy()*(x-y)
            Y[i+n0+1,:]=y+dt*(f(y)-u)+np.sqrt(dt)*z[i]*(1*np.sum(np.sin(2*(x-y)))-1*np.sum(np.sin(x_t-y_t))) *((1+i*dt))*5
            U[i,:]= u
            if i%10000==0:
                print('NN:',np.min(x-y),np.min(x_t-y_t),np.max(u))
    return X,Y,U

def drive(n,dt):
    a,b,c,d,B,m0,m1=7,0.35,0.5,7,1,-1/7,-40/7
    # x0 = np.random.uniform(-1,1,3)
    x0 = np.array([1.5,-4.4,0.15])
    X = np.zeros([n,3])
    X[0,:] = x0

    for i in range(n-1):
        x = X[i,:]
        dx = f(x)
        X[i+1,:] = x+dx*dt
    return X



fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
np.random.seed(3)
'''
generate
'''

m = 10
n = 50000
dt = 0.00001

X2,Y2,U2=run(50000,0.00001,3,3)
print(X2.shape)
plt.plot(np.arange(len(X2)),X2[:,1])
plt.plot(np.arange(len(Y2)),Y2[:,1])

# X,Y,Z = np.zeros([m,6000,3]),np.zeros([m,6000,3]),np.zeros([m,6000,3])
# for i in range(m):
#     X1,Y1,U1=run(100000,0.000005,1,20*i)
#     X2,Y2,U2=run(100000,0.000005,3,20*i)
#     X[i,:] = X1[0:120000:20,:]
#     Y[i,:] = Y1[0:120000:20,:]
#     Z[i,:] = Y2[0:120000:20,:]
#     print(i)
# np.save('data_vary.npy',{'drive':X,'LC':Y,'NDC':Z})

# data = np.load('data_vary.npy',allow_pickle=True).item()
# X,Y,Z=data['drive'],data['LC'],data['NDC']
# print(X.shape)
# plt.plot(np.arange(6000),X[0,:,1],label='drive')
# plt.plot(np.arange(6000),Y[0,:,1],label='LC')
# plt.plot(np.arange(6000),Z[0,:,1],label='NDC')
# plt.legend()

# X1,Y1,U1 = run(50000,0.00001,1)
# X2,Y2,U2 = run(50000,0.00001,2)
# np.save('original.npy',{'drive':X1[0:200000:20,:],'response':Y1[0:200000:20,:]})
# np.save('orbit.npy',{'drive':X1[0:100000:10,:],'LC':Y1[0:10000:10,:],'NDC':Y2[0:10000:10,:]})
# np.save('time_invary.npy',{'drive':X1[0:50000:5,:],'LC':Y1[0:50000:5,:],'NDC':Y2[0:50000:5,:]})
# np.save('time_vary.npy',{'drive':X1[0:100000:10,:],'LC':Y1[0:100000:10,:],'NDC':Y2[0:50000:5,:]}) $ length=10000




# data=np.load('time_invary.npy',allow_pickle=True).item()
# X,Y,Z=data['drive'],data['LC'],data['NDC']
# print(X.shape,Y.shape,Z.shape)


# ax.plot(X1[0:100000:10,0],X1[0:100000:10,1],X1[0:100000:10,2])
# ax.plot(Y1[0:100000:10,0],Y1[0:100000:10,1],Y1[0:100000:10,2])
# plt.plot(np.arange(len(X1)),X1[:,1])
# plt.plot(np.arange(len(X1)),Y1[:,1])
n = 200000
dt= 0.0001
def plot():
    ax1 = fig.add_subplot(141, projection='3d')
    data = np.load('original.npy',allow_pickle=True).item()
    X = data['drive']
    Y = data['response']
    ax1.plot(X[:,0],X[:,1],X[:,2],label='drive')
    ax1.plot(Y[:,0],Y[:,1],Y[:,2],label='response')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    ax1.set_zlabel(r'$z$')
    # a = np.linspace(0,dt*(len(U)-1),len(U))
    # energy = integrate.trapz(np.array(np.sum(U**2,axis=1)),a)
    plt.legend()

    ax2 = fig.add_subplot(142, projection='3d')
    data=np.load('orbit.npy',allow_pickle=True).item()
    X,Y,Z = data['drive'],data['LC'],data['NDC']
    ax2.plot(X[:,0],X[:,1],X[:,2],label='drive')
    ax2.plot(Y[:,0],Y[:,1],Y[:,2],label='LC')
    ax2.plot(Z[:,0],Z[:,1],Z[:,2],label='NDC')
    plt.legend()


    plt.subplot(143)
    data=np.load('time_invary.npy',allow_pickle=True).item()
    X,Y,Z = data['drive'],data['LC'],data['NDC']
    plt.plot(np.arange(len(X)),X[:,1],label='drive')
    plt.plot(np.arange(len(X)),Y[:,1],label='LC')
    plt.plot(np.arange(len(X)),Z[:,1],label='NDC')
    plt.xticks([0,5000,10000],[0,0.25,0.5])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y$')
    plt.legend()
    plot_grid()

    plt.subplot(144)
    data=np.load('time_vary.npy',allow_pickle=True).item()
    X,Y,Z = data['drive'],data['LC'],data['NDC']
    plt.plot(np.arange(len(X)),X[:,1],label='drive')
    plt.plot(np.arange(len(X)),Y[:,1],label='LC')
    plt.plot(np.arange(len(X)),Z[:,1],label='NDC')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y$')
    plt.xticks([0,5000,10000],[0,0.05,0.1])
    plt.legend()
    plot_grid()

# plot()
plt.show()
end=timeit.default_timer()
print(end-start)
