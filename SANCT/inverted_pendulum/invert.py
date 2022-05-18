import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
from scipy.integrate import odeint
from cvxopt import solvers,matrix
from functions import *


model = ControlNet(4,16,2)

def f(state):
    x, y = state
    G = 9.81  # gravity
    L = 0.5   # length of the pole
    m = 0.15  # ball mass
    b = 0.1   # friction
    return np.array([y, G*np.sin(x)/L +(-b*y)/(m*L**2)])

def g(state):
    x,y = state
    return np.array([np.sin(x),np.sin(y)])

def initial(t):
    w = np.pi/2 - t
    v = -1.0 + t/3
    return np.array([w,v])

def qp(inp1,inp2,epi=0.2,p=20.0):
    def h(x):
        M = 6
        return M-x**2
    # x,y,xt,yt=np.float64(x),np.float64(y),np.float64(xt),np.float64(yt)
    x,y = inp1
    xt,yt=inp2
    gamma = 5.0
    g = 9.81  # gravity
    L = 0.50   # length of the pole
    m = 0.15  # ball mass
    b = 0.1   # friction
    P = matrix(np.diag([2.0,2.0,2.0*p,2.0*p]))
    q = matrix([0.0,0.0,0.0,0.0])
    G = matrix(np.array([[2*x/h(x)**2,0.0,-1.0,0.0],[x,y,0.0,-1.0]]))
    h = matrix([gamma*h(x)-np.cos(xt)**2*(1/h(x)**2+4*x**2/h(x)**3)-2*x*y/h(x)**2,
                b*y**2/(m*L**2)-y*g*np.sin(x)/L-x*y-(np.sin(xt)**2+np.sin(yt)**2)/2-(x**2+y**2)/(2*epi)+(xt**2+yt**2)/(2*epi+0.1)]) # 在Lie算子里加入V/epi项
    # b*y**2/(m*L**2)-y*g*np.sin(x)/L-x*y-(np.cos(xt)**2+np.cos(yt)**2)/2-(x**2+y**2)/(2*epi)
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h)  # 调用优化函数solvers.qp求解
    u =np.array(sol['x'])
    return u

def run_0(n,dt,case,seed):
    np.random.seed(seed)
    tau = 0.1
    epi = 1e-3
    nd = int(tau/dt)
    X = np.zeros([n+nd,2])
    DU = np.zeros([n,2])
    SU = np.zeros([n,2])
    z = np.random.normal(0,1,n)
    for i in range(nd+1):
        t=i*dt-tau
        X[i,:] = initial(t)
    for i in range(n-1):
        x = X[i+nd,:]
        x_t = X[i,:]
        df = f(x)
        dg = g(x_t)
        if case == 0:
            X[i+nd+1,:] = x+df*dt+np.sqrt(dt)*z[i]*dg
        if case == 'qp':
            u1,u2,d1,d2 = qp(x,x_t)
            du = np.array([u1[0],u2[0]])
            X[i+nd+1,:]=x+(df+du)*dt+np.sqrt(dt)*z[i]*dg
        if case == 'S':
            # if np.sum(x**2)<epi:
            #     u=np.zeros_like(x)
            # else:
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((x,x_t))).to(torch.float32).unsqueeze(0)
                u = model(input).detach().numpy()
            X[i+nd+1,:]=x+df*dt+np.sqrt(dt)*z[i]*(dg+u)
        #     SU[i,:] = u
    return X,DU,SU


seed = 81 #1,4,63,80,81
n = 50000
dt = 0.00001
tau = 0.1
m = 10
font_size = 25
# model.load_state_dict(torch.load('Safe_S_1000.pkl'))
# model.load_state_dict(torch.load('S_1000.pkl'))
# X,DU,SU=run_0(n,dt,'S',seed)
# print(X.shape)
# plt.plot(np.arange(len(X)),X[:,0],color=colors[2])
'''
generate
'''
# model.load_state_dict(torch.load('Safe_S_1000.pkl'))
model.load_state_dict(torch.load('S_1000.pkl'))
data = np.zeros([5,6000,2])
seed = [1,4,63,80,81]
for i in range(5):
    X,DU,SU=run_0(n,dt,'qp',seed[i])
    print(X.shape)
    data[i,:] = X[0:60000:10,:]
    print(i)
np.save('qp.npy',data)


# plt.subplot(141)
# model.load_state_dict(torch.load('safe_S.pkl'))
# X,DU,SU=run_0(2000,0.001,0,seed)
# plt.plot(np.arange(len(X)),X[:,0],color=colors[2])
# plt.xticks([0,1000,2000],[0,1.0,2.0])
# plt.xlabel(r'$t$',fontsize=font_size)
# plt.ylabel(r'$\theta$',fontsize=font_size)
# plot_grid()
#
# plt.subplot(142)
# model.load_state_dict(torch.load('safe_S.pkl'))
# X,DU,SU=run_0(n,dt,'qp',seed)
# plt.plot(np.arange(len(X)),X[:,0],color=colors[2])
# plt.xticks([0,10000,20000],[0,0.5,1.0])
# plt.xlabel(r'$t$',fontsize=font_size)
# plot_grid()
#
# plt.subplot(143)
# model.load_state_dict(torch.load('Safe_S_2000_200.pkl'))
# model.load_state_dict(torch.load('S_2000.pkl'))
# X,DU,SU=run_0(n,dt,'S',3)
# print(X.shape)
# X = X[2000:10000,:]
# plt.plot(np.arange(len(X)),X[:,0],color=colors[0])
# plt.plot(np.arange(len(X)),X[:,1],color=colors[1])
# plt.xticks([0,10000,20000],[0,0.5,1.0])
# plt.xlabel(r'$t$',fontsize=font_size)
# plot_grid()
#
# plt.subplot(144)
# model.load_state_dict(torch.load('S.pkl'))
# X,DU,SU=run_0(n,dt,'S',seed)
# plt.plot(np.arange(len(X)),X[:,0],color=colors[2])
# plt.xticks([0,10000,20000],[0,0.5,1.0])
# plt.xlabel(r'$t$',fontsize=font_size)
# plot_grid()

plt.show()