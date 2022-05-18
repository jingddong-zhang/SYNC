import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit

colors = [
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 248/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
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

linewidth = 0.5
fontsize = 15
legend_loc = "lower right"
def plot_grid():
    # lw_major, lw_minor = 1.5, 1
    lw_major, lw_minor = 0.3, 0.2
    plt.grid(b=True, which='major', color='gray', alpha=0.3, linestyle='dashdot', lw=lw_major)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.4, ls='-', lw=lw_minor)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)


'''
Description of data: 5 sample trajectories, (:,:,0) represent theta component
'uncontrol.npy': (5,5000,2), (:,:1000,:) relates to initial (-0.1,0),(:,1000:,:) relates to (0,0.4)
'qp.npy': the same as 'uncontrol.npy'

'nsc.npy': (5,60000,2), (:,:1000,:) relates to initial (-0.1,0),(:,1000:,:) relates to (0,0.5)
'nsc_safe.npy':the same as 'nsc.npy'
'''

font_size = 25

def subplot(X,xticks1,xticks2,title): # plot y component of chua's model
    alpha = 0.6
    mean_x,std_x=np.mean(X[:,:,0],axis=0),np.std(X[:,:,0],axis=0)
    length = len(mean_x)
    plt.fill_between(np.arange(length),mean_x-std_x,mean_x+std_x,color=colors[0],alpha=alpha)
    plt.plot(np.arange(length),mean_x,color=colors[0],linewidth=linewidth)
    # plt.xticks([0,250,1250,2250],[-0.1,0,0.4,0.8])
    plt.xlabel(r'$t$', fontsize=fontsize)
    plt.xticks(xticks1,xticks2,fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plot_grid()

def plot():
    plt.subplot(141)
    X=np.load('uncontrol.npy',allow_pickle=True)
    subplot(X,[0,1000,3000,5000],[-0.1,0,0.2,0.4],'Without control')
    plt.ylabel(r'$\theta$', fontsize=fontsize)
    plt.yticks([1, 1.5, 2, 2.5, 3], [1, "", 2, "", 3])
    plt.tick_params(labelsize=fontsize)

    plt.subplot(142)
    X=np.load('qp.npy',allow_pickle=True)
    subplot(X,[0,1000,3000,5000],[-0.1,0,0.2,0.4],'Baseline')
    plt.yticks([0, 0.5, 1, 1.5, 2], [0, "", 1, "", 2])
    plt.tick_params(labelsize=fontsize)


    color_fill = [255/255, 255/255, 240/255]
    color_axhline = [0/255, 120/255, 0/255]
    plt.subplot(143)
    X=np.load('nsc.npy',allow_pickle=True)
    X=X[:,0:60000:3,:]
    plt.fill_between(np.arange(X.shape[1]),-2*np.pi,2*np.pi,color=color_fill,alpha=1.0)
    subplot(X,[0,10000/3,10000,20000],[-0.1,0,0.2,0.4],'NSC')


    plt.axhline(y=2*np.pi,ls="--",color=color_axhline,lw=linewidth, alpha=0.5)
    plt.axhline(y=-2*np.pi,ls="--",color=color_axhline,lw=linewidth, alpha=0.5)

    plt.yticks([-5, 0, 5, 10, 15], [-5, "", 5, "", 15])
    plt.tick_params(labelsize=fontsize)

    plt.subplot(144)
    X=np.load('nsc_safe.npy',allow_pickle=True)
    X=X[:,0:60000:10,:]
    plt.fill_between(np.arange(X.shape[1]),-2*np.pi,2*np.pi,color=color_fill,alpha=1.0)
    subplot(X,[0,1000,3000,5000],[-0.1,0,0.2,0.4],'NSC+Safe')
    # subplot(X,[0,10000,30000,50000],[-0.1,0,0.2,0.4],'NSC+Safe')

    plt.axhline(y=2*np.pi,ls="--",color=color_axhline,lw=linewidth, alpha=0.5)
    plt.axhline(y=-2*np.pi,ls="--",color=color_axhline,lw=linewidth, alpha=0.5)
    plt.yticks([-5, -2.5, 0, 2.5, 5], [-5, "", 0, "", 5])
    plt.tick_params(labelsize=fontsize)

    print('X shape:{}'.format(X.shape))

plot()

plt.show()


