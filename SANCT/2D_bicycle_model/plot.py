import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import timeit

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


'''
Description of data of kinematic bicycle model
All the data are saved as dictionary with three keys 'X','DU','SU', in Figure 4 only 'X' is used
'uncontrol.npy': 'X': (10,6000,4), 10 represent 10 sample trajectories, (:,:,0) is x, (:,:,1) is y
      (:,:1000,:) represent initial data in (-0.1,0), (:,1000:,:) represent controlled data in (0,0.5)

'ndc.npy': NDC control, 'X':(10,31000,4),
      (:,:1000,:) represent initial data in (-0.1,0), (:,1000:,:) represent controlled data in (0,0.5)

'qp.npy': QP control, data shape is the same as 'ndc.npy'
 'S.npy': NSC,   data shape is the same as 'ndc.npy'
 'D.npy': NSC+D, data shape is the same as 'ndc.npy'
 'M.npy': NSC+M, data shape is the same as 'ndc.npy'
'''

font_size = 20

# plot the x,y coordinate along time
def subplot(X,xticks1,xticks2,yticks1,yticks2,ylim,title):
    alpha = 0.5
    # calculate mean and std of coordinate x and y
    mean_x,std_x,mean_y,std_y=np.mean(X[:,:,0],axis=0),np.std(X[:,:,0],axis=0),np.mean(X[:,:,1],axis=0),np.std(X[:,:,1],axis=0)
    length = len(mean_x)
    plt.fill_between(np.arange(length),mean_x-std_x,mean_x+std_x,color=colors[0],alpha=alpha)
    plt.plot(np.arange(length),mean_x,color=colors[0],label=r'$x$')
    plt.fill_between(np.arange(length),mean_y-std_y,mean_y+std_y,color=colors[1],alpha=alpha)
    plt.plot(np.arange(length),mean_y,color=colors[1],label=r'$y$')
    plot_grid()
    plt.legend(fontsize=font_size)
    plt.xticks(xticks1,xticks2,fontsize=font_size)
    plt.yticks(yticks1,yticks2,fontsize=font_size)
    plt.ylim(ylim)
    plt.title('{}'.format(title),fontsize=font_size)


def plot():
    plt.subplot(231)
    data = np.load('uncontrol.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    print(X.shape)
    subplot(X,[0,1000,3500,6000],[-0.1,0,0.25,0.5],[-2,0,2,4],[-2,0,2,4],[-2,5],'Uncontrol')
    plt.ylabel('state variables',fontsize=font_size)

    plt.subplot(232)
    data = np.load('ndc.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    X = X[:,0:31000:10,:]
    subplot(X,[0,1100,2100,3100],[-0.1,1,2,3],[0,1,2],[0,'',2],[-0.2,2.5],'NDC')

    plt.subplot(233)
    data = np.load('qp.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    X = X[:,0:31000:10,:]
    subplot(X,[0,1100,2100,3100],[-0.1,1,2,3],[0,1,2,3],[0,1,2,3],[-0.2,2.5],'QP')

    plt.subplot(234)
    data = np.load('S.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    subplot(X,[0,1000,3500,6000],[-0.1,0,0.25,0.5],[0,1,2,3],[0,1,2,3],[-0.2,2.5],'NSC')
    plt.ylabel('state variables',fontsize=font_size)
    plt.xlabel(r'$t$',fontsize=font_size)

    plt.subplot(235)
    data = np.load('D.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    subplot(X,[0,1000,3500,6000],[-0.1,0,0.25,0.5],[0,1,2,3],[0,1,2,3],[-0.2,2.5],'NSC+D')
    plt.xlabel(r'$t$',fontsize=font_size)

    plt.subplot(236)
    data = np.load('M.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    X = np.delete(X[:,0:60000:10,:],[2,4],axis=0)  # delete the divergence trajectory due to big dt in euler method
    subplot(X,[0,1000,3500,6000],[-0.1,0,0.25,0.5],[0,1,2,3],[0,1,2,3],[-0.2,2.5],'NSC+M')
    plt.xlabel(r'$t$',fontsize=font_size)

plot()
plt.show()