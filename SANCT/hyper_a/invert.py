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
            SU[i,:] = u
    return X,DU,SU


seed = 4 #1,4,63,80,81
n = 50000
dt = 0.00001
tau = 0.1
m = 10
font_size = 25
theta = 0.1
# model.load_state_dict(torch.load('./data/S_{}.pkl'.format(theta)))
# model.load_state_dict(torch.load('S_1000.pkl'))
# X,DU,SU=run_0(n,dt,'S',seed)
# print(X.shape)
# plt.plot(np.arange(len(X)),X[:,0],color=colors[2])
# plt.plot(np.arange(len(X)),X[:,1],color=colors[7])

'''
generate
'''
# start = timeit.default_timer()
# data_x = np.zeros([19,12,60000,2]) # alpha,seed,n,dt
# data_u = np.zeros([19,12,50000,2])
# for k in range(19):
#     theta=float(format(k*0.05+0.05,'.2f'))
#     model.load_state_dict(torch.load('./data/S_{}.pkl'.format(theta)))
#     for i in range(12):
#         X,DU,SU=run_0(n,dt,'S',i*5+1)
#         data_x[k,i,:,:] = X
#         data_u[k,i,:,:] = SU
#         print(k,i,timeit.default_timer()-start)
# np.save('./data/data_X',data_x) # (0,3),(3,3)
# np.save('./data/data_U',data_u)


data_qp = np.zeros([12,60000,2])
for i in range(12):
    X,DU,SU=run_0(n,dt,'qp',i)
    # X,DU,SU=run_0(n,dt,'qp',i*5+1)
    data_qp[i,:,:]=X
    print(i)
np.save('./data/data_qp',data_qp)


plt.show()