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
W = np.load('./data/topo_W.npy',allow_pickle=True)
sol = np.load('./data/solution.npy')

def f(x):
    x = np.expand_dims(x,axis=0)
    return -x+np.matmul(x**2/(1+x**2),W.T)

def g(x):
    return np.sin(x/sol*np.pi)

def initial(x0,t):
    return x0+np.random.normal(0,1,len(x0))*np.sqrt(t)

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

def run_0(n,dt,case,seed,delta):
    np.random.seed(seed)
    x0 = np.random.uniform(0,delta,100)
    tau = 0.5
    epi = 1e-3
    nd = int(tau/dt)
    X = np.zeros([n+nd,100])
    DU = np.zeros([n,100])
    SU = np.zeros([n,100])
    z = np.random.normal(0,1,n)
    X[0,:]=x0
    for i in range(nd):
        x = X[i,:]
        X[i+1,:] = x+np.random.normal(0,delta,len(x))*np.sqrt(dt)
        # X[i+1,:] =
    for i in range(n-1):
        x = X[i+nd,:]
        x_t = X[i,:]
        df = f(x)
        dg = g(x_t)
        if case == 0:
            X[i+nd+1,:] = x+df*dt+np.sqrt(dt)*z[i]*dg
        if case == 1:
            X[i+nd+1,:] = x+df*dt
        if case == 'qp':
            u1,u2,d1,d2 = qp(x,x_t)
            du = np.array([u1[0],u2[0]])
            X[i+nd+1,:]=x+(df+du)*dt+np.sqrt(dt)*z[i]*dg
        if case == 'D':
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((x,x_t))).to(torch.float32).unsqueeze(0)
                u = modeld(input).detach().numpy()
            X[i+nd+1,:]=x+(df+u)*dt+np.sqrt(dt)*z[i]*(dg)
            if i%500==0:
                print(np.mean(x),np.mean(u))
            DU[i,:] = u
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
model = ControlNet(200,800,100)
model.load_state_dict(torch.load('./data/S_0.8.pkl'))
modeld = ControlNet(200,800,100)
modeld.load_state_dict(torch.load('./data/ES.pkl'))

def generate1():
    end = np.zeros(21)
    for k in range(21):
        delta=float(format(k*0.05,'.2f'))
        X,DU,SU = run_0(10000,0.001,0,0,delta)
        end[k]=np.mean(X[-1,:])
        print(k)
    np.save('./data/end',end)

def generate2(delta,N=10):
    data = np.zeros([N,25000,100])
    for k in range(N):
        X,DU,SU=run_0(20000,0.0001,'S',k,delta)
        data[k,:,:]=X
        print(k)
    return data

data = np.zeros([2,10,25000,100])
data[0,:]=generate2(0.25)
data[1,:]=generate2(1)
np.save('./data/control2',data)

# data = np.load('./data/control2.npy',allow_pickle=True)
# data = np.delete(data[:,:,0:15000:10,:],[3],axis=1)
# data = np.mean(data,axis=3)
# x1 = np.mean(data[0,:],axis=0)
# print(x1.shape)

# X,DU,SU = run_0(10000,0.0001,'S',2,1.0)
# print(X.shape,np.mean(X[-1,:]))
# X=X[-10000:,:]
# # np.save('./data/control',X)
# plt.plot(np.arange(len(X)),np.mean(X,axis=1))
# plt.axhline(y=np.mean(sol), c="r", ls="--", lw=2)




plt.show()