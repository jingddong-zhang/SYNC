from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import pickle
from functions import *
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from networkx.generators.random_graphs import barabasi_albert_graph, erdos_renyi_graph
import pandas as pd

font_size=20
def subplot(ax,zd,coff=colors[1]):
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    ax.view_init(24,40) #rotate
    # ax.set_xticks([-4,-2,0, 2,4],['','','','',''])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_zlim(0,10)
    ax.set_zticks([0,5,10])


    z = np.linspace(0,2*np.pi,100)
    x = 5*np.sin(z)
    y = 5*np.cos(z)
    ax.scatter3D(x,y,zd,c=zd,cmap='jet',norm=mpl.colors.Normalize(vmin=0, vmax=10))

    # for i in range(100):
    #     c=zd[i]/10
    #     ax.scatter3D(x[i:i+1],y[i:i+1],zd[i:i+1],'.',color=[c*coff[0],c*coff[1],c*coff[2]])

'''
uncontrol.npy: (5500,100)表示100个振子长度为5500的轨道，0到500表示初值，500到5000表示受控轨道
control.npy：(10000,100)表示100个振子长度为10000的轨道，0到1000表示初值，1000到10000表示受控轨道
'''
Z = np.load('./data/control.npy')
print(Z.shape)
def plot1():
    Z = np.load('./data/uncontrol.npy')
    fig = plt.figure()
    ax1 = fig.add_subplot(321,projection='3d')
    subplot(ax1,Z[0,:])
    plt.title('Original',fontsize=font_size)
    ax2 = fig.add_subplot(323,projection='3d')
    subplot(ax2,Z[2500,:])
    ax3 = fig.add_subplot(325,projection='3d')
    subplot(ax3,Z[-1,:])

    Z = np.load('./data/control.npy')
    ax4 = fig.add_subplot(322,projection='3d')
    subplot(ax4,Z[0,:])
    plt.title('Controlled',fontsize=font_size)
    ax5 = fig.add_subplot(324,projection='3d')
    subplot(ax5,Z[7500,:])
    ax6 = fig.add_subplot(326,projection='3d')
    subplot(ax6,Z[-1,:])
# plot1()

def plot2():
    plt.subplot(131)
    W=np.load('./data/topo_W.npy',allow_pickle=True)
    G=nx.Graph(W)
    pos=nx.circular_layout(G)
    nodecolor=G.degree()  # 度数越大，节点越大，连接边颜色越深
    nodecolor2=pd.DataFrame(nodecolor)  # 转化称矩阵形式
    nodecolor3=nodecolor2.iloc[:,1]  # 索引第二列
    edgecolor=range(G.number_of_edges())  # 设置边权颜色
    print(edgecolor)
    nx.draw(G,pos,with_labels=False,node_size=nodecolor3*12,node_color=nodecolor3*15,edge_color=edgecolor,
            cmap=plt.cm.jet)

    plt.subplot(132)
    end = np.load('./data/end.npy',allow_pickle=True)
    plt.scatter(np.arange(len(end)),end,s=105,c=end,marker='.',alpha=1.99,cmap='rainbow')
    plt.xlabel(r'$\sigma$',fontsize=font_size)
    plt.ylabel(r'$\bar{x}$',fontsize=font_size)
    plt.xticks([0,5.0,10.0,15.0,20.0],[0,0.25,0.5,0.75,1.0],fontsize=font_size)
    plt.yticks([0,1.5,3.0],fontsize=font_size)
    plt.text(3,0.2,r'$\mathbf{x}_0^\ast$',fontsize=font_size,color='#ae7df1')
    plt.text(15,3.2,r'$\mathbf{x}_1^\ast$',fontsize=font_size,color='r')
    plot_grid()

    plt.subplot(133)
    alpha=0.5
    data = np.load('./data/control2.npy',allow_pickle=True)
    data = np.delete(data[:,:,0:15000:10,:],[3],axis=1)  # delete the divergence trajectory due to big dt in euler method
    data = np.mean(data,axis=3)
    mean_1,std_1,mean_2,std_2 = np.mean(data[0,:],axis=0),np.std(data[0,:],axis=0),np.mean(data[1,:],axis=0),\
                    np.std(data[1,:],axis=0)
    length = len(mean_1)
    plt.fill_between(np.arange(length),mean_1-std_1,mean_1+std_1,color=colors[0],alpha=alpha)
    plt.plot(np.arange(length),mean_1,color=colors[0],label=r'$\sigma=0.25$')
    plt.fill_between(np.arange(length),mean_2-std_2,mean_2+std_2,color=colors[1],alpha=alpha)
    plt.plot(np.arange(length),mean_2,color=colors[1],label=r'$\sigma=1$')
    plt.xticks([0,500,1500],[-0.5,0,1.0],fontsize=font_size)
    plt.yticks([-0.5,0,0.5,1.0],fontsize=font_size)
    plt.ylim(-0.2,1.2)
    plt.legend(fontsize=font_size)
    plt.xlabel('Time',fontsize=font_size)
    plt.ylabel(r'$\bar{x}$',fontsize=font_size)
    plot_grid()
# plot2()
# ax1.plot3D(x,y,z,'gray')    #绘制空间曲线

plt.show()