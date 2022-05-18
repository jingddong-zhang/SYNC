import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import timeit
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pickle

colors = [
    [233/256,	110/256, 236/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    [255/255, 165/255, 0],
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [107/256,	161/256,255/256], # #6ba1ff
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
]
colors = np.array(colors)

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass


def subplot(X,Y,Z): # plot y component of chua's model
    alpha = 0.3
    length = 6000
    mean_x,std_x,mean_y,std_y,mean_z,std_z = np.mean(X[:,:,1],axis=0),np.std(X[:,:,1],axis=0),np.mean(Y[:,:,1],axis=0),\
                    np.std(Y[:,:,1],axis=0),np.mean(Z[:,:,1],axis=0),np.std(Z[:,:,1],axis=0)
    plt.fill_between(np.arange(length),mean_x-std_x,mean_x+std_x,color=colors[0],alpha=alpha)
    plt.plot(np.arange(length),mean_x,color=colors[0],label='drive')
    plt.fill_between(np.arange(length),mean_y-std_y,mean_y+std_y,color=colors[1],alpha=alpha)
    plt.plot(np.arange(length),mean_y,color=colors[1],label='LC')
    plt.fill_between(np.arange(length),mean_z-std_z,mean_z+std_z,color=colors[2],alpha=alpha)
    plt.plot(np.arange(length),mean_z,color=colors[2],label='NDC')
    plt.xticks([0,1000,3500,6000],[-0.1,0,0.25,0.5])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y$')
    plt.legend()
    plot_grid()

'''
Description of data of 3-D chua's model
All the data are saved as dictionary
'original.npy': used for orbit plot, uncontrol data, two keys 'drive': (10000,3), 'response':(10000,3) 
'orbit.npy': used for orbit plot, controlled by LC/NDC, three keys 'drive': (10000,3), 'LC':(1000,3), 'NDC':(1000,3)

'data_invary.npy': used for trajectories plot, controlled by LC/NDC, 
      three keys 'drive': (10,6000,3), 'LC':(10,6000,3), 'NDC':(10,6000,3), 10 represent 10 sample trajectories,
      (:,:1000,:) represent initial data in (-0.1,0), (:,1000:,:) represent controlled data in (0,0.5)

 'data_vary.npy': used for trajectories plot in time-varying case, data shape and meaning are the same as above. 
'''


def plot():
    fig=plt.figure()
    ax1 = fig.add_subplot(141, projection='3d')
    data = np.load('original.npy',allow_pickle=True).item()
    X = data['drive']
    Y = data['response']
    ax1.plot(X[:,0],X[:,1],X[:,2],label='drive')
    ax1.plot(Y[:,0],Y[:,1],Y[:,2],label='response')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    ax1.set_zlabel(r'$z$')
    plt.legend()

    ax2 = fig.add_subplot(142, projection='3d')
    data=np.load('orbit.npy',allow_pickle=True).item()
    X,Y,Z = data['drive'],data['LC'],data['NDC']
    print(Z.shape)
    ax2.plot(X[:,0],X[:,1],X[:,2],label='drive')
    ax2.plot(Y[:,0],Y[:,1],Y[:,2],label='LC')
    ax2.plot(Z[:,0],Z[:,1],Z[:,2],label='NDC')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    ax2.set_zlabel(r'$z$')
    plt.legend()

    plt.subplot(143)
    data=np.load('data_invary.npy',allow_pickle=True).item()
    X,Y,Z = data['drive'],data['LC'],data['NDC']
    subplot(X,Y,Z)

    plt.subplot(144)
    data=np.load('data_vary.npy',allow_pickle=True).item()
    X,Y,Z = data['drive'],data['LC'],data['NDC']
    subplot(X,Y,Z)

# plot()
plt.show()

