import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from networkx.generators.random_graphs import barabasi_albert_graph, erdos_renyi_graph
from scipy.integrate import odeint
import os
import torch
import pandas as pd

def generate_A(N=100,neibor=4,p=0.7):
    '''
    :param shape: 矩阵A的shape  （D_r, D_r）
    :param D_r:   矩阵A的维度
    :return:  生成后的矩阵A
    '''
    # G = erdos_renyi_graph(D_r, 4 / D_r, seed = seed)   # 生成ER图，节点为D_r个，连接概率p = 3 /D_r
    G = nx.random_graphs.watts_strogatz_graph(N,neibor,p)
    # pos=nx.circular_layout(G)
    # nx.draw(WS, pos, with_labels = False, node_size = 30)
    degree = [val for (node, val) in G.degree()]
    print('平均度:', sum(degree) / len(degree))
    G_A = nx.to_numpy_matrix(G)  # 生成后的图转化为邻接矩阵A， 有边的为1，无边为0
    return G_A

def initial_W(shape, lower,upper):
    return np.random.uniform(lower, upper, size=shape)

def weight_A(G_A,w):
    index = np.where(G_A > 0)  # 找出有边的位置
    res_A = np.zeros_like(G_A)
    res_A[index] = initial_W([len(index[0]), ],0, 2*w)  # 对有边的位置按均匀分布[0, a]进行随机赋值
    return res_A


D_r  = 100
A = generate_A(100,4,0.5)
W = weight_A(A,1.0)
print(W.shape)
# np.save('./data/topo_W',W)
G = nx.Graph(A)
pos=nx.circular_layout(G)
# pos=nx.spring_layout(G)
# nx.draw(G,pos)


nodecolor=G.degree() #度数越大，节点越大，连接边颜色越深
nodecolor2=pd.DataFrame(nodecolor) #转化称矩阵形式
nodecolor3=nodecolor2.iloc[:,1] #索引第二列
edgecolor=range(G.number_of_edges()) #设置边权颜色
print(edgecolor)
nx.draw(G, pos, with_labels=False,node_size=nodecolor3*12,node_color=nodecolor3*15,edge_color=edgecolor,cmap=plt.cm.jet)
plt.show()

