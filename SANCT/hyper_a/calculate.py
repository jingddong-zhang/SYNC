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
font_size= 25
tick_size= 20
data = np.load('./data/data_X.npy',allow_pickle=True) # (0,3),(3,3)
data_u = np.load('./data/data_U.npy',allow_pickle=True)
print(data.shape)
end = np.zeros([19])
ene = np.zeros([19])
for r in range(19):
    end[r] = np.abs(np.mean(data[r,:,-1,0]))
    ene[r] = energy(data_u[r,:])
print(end.shape,ene)
print(end)
plt.subplot(121)
plt.scatter(np.arange(len(end)),end, s=105, c=end, marker='.',alpha=1.99,cmap='rainbow')
plot_grid()
plt.xlabel(r'$\alpha$',fontsize=font_size)
plt.ylabel(r'$\theta$',fontsize=font_size)
plt.xticks([-1.0,  3.0, 7.0, 11.0, 15.0, 19.0],[0,0.2,0.4,0.6,0.8,1.0],fontsize=tick_size)
plt.yticks([0,0.006,0.012],fontsize=tick_size)
cb=plt.colorbar()
cb.ax.tick_params(labelsize=16)

plt.subplot(122)
plt.scatter(np.arange(len(end)),ene, s=105, c=ene, marker='.',alpha=1.99,cmap='rainbow')
plt.xlabel(r'$\alpha$',fontsize=font_size)
plt.ylabel(r'$\rm{Energy}$',fontsize=font_size)
plt.xticks([-1.0,  3.0, 7.0, 11.0, 15.0, 19.0],[0,0.2,0.4,0.6,0.8,1.0],fontsize=tick_size)
plt.yticks([13,16,19],fontsize=tick_size)
plot_grid()
# # plt.axvline(7.5,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3)
# plt.axvline(11.5,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3)
# plt.axhline(0.0,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3)
# plt.yticks([0,0.03,0.06])
cb=plt.colorbar()
cb.ax.tick_params(labelsize=16)
plt.show()

