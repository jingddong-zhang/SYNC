import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from scipy import integrate
from functions import *


def energy(U,n=50000,dt=0.00001):
    a=np.linspace(0,dt*(n-1),n)
    e = 0.0
    for i in range(len(U)):
        e += integrate.trapz(np.array(np.sum(U[i,:]**2,axis=1)),a)
    return e/float(len(U))
'''
Calculate and plot the mean end position of trajectories under learning control with each $alpha$ 
'''
font_size= 20
data = np.load('./data/data_X.npy',allow_pickle=True)
data_qp = np.load('./data/data_qp.npy',allow_pickle=True)
print(data.shape)
data = data[:,:,0:60000:10,:]
data_qp = data_qp[:,0:60000:10,:]
a1 = data[2,:]
a2 = data[10,:]
a3 = data[18,:]
print(a1.shape)
length=6000
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.plot(np.arange(length),a1[i,:,0],color=colors[0],label=r'$\alpha=0.1$')
    plt.plot(np.arange(length),a2[i,:,0],color=colors[1],label=r'$\alpha=0.5$')
    plt.plot(np.arange(length),a3[i,:,0],color=colors[2],label=r'$\alpha=0.9$')
    plt.plot(np.arange(length),data_qp[i,:,0],color=colors[-2],label=r'Baseline')
    plt.legend()
    plt.xticks([0,1000,3500,6000],[-0.1,0,0.25,0.5])
    plt.xlabel('Time')
    plt.ylabel(r'$\theta$')
    plt.title('Random seed {}'.format(i))
plt.show()

