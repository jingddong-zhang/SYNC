import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
from scipy.integrate import odeint
from functions import *
from cvxopt import solvers,matrix

def qp(inp1,inp2,epi=0.2,p=10.0):
    px,py,w,v=inp1
    pxt,pyt,wt,vt=inp2
    gamma = 5.0
    P = matrix(np.diag([2.0,2.0,2.0,2.0,2.0*p]))
    q = matrix([0.0,0.0,0.0,0.0,0.0])
    G = matrix(np.array([[px,py,w,v,-1.0]]))
    h = matrix([-(pxt**2+pyt**2)/2-(px**2+py**2+w**2+v**2)/(2*epi)-px*v*np.cos(w)-py*v*np.sin(w)-w*v-v*(px**2+py**2)]) # 在Lie算子里加入V/epi项
    # b*y**2/(m*L**2)-y*g*np.sin(x)/L-x*y-(np.cos(xt)**2+np.cos(yt)**2)/2-(x**2+y**2)/(2*epi)
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h)  # 调用优化函数solvers.qp求解
    u =np.array(sol['x'])
    return u

def f(x,u=0):
    px,py,w,v=x
    return np.array([v*np.cos(w),v*np.sin(w),v,px**2+py**2])

def g(y,u=0):
    px,py,w,v=y
    return np.array([px,py,0.0,0.0])

def initial(t):
    w = np.pi/2+t
    v = 2+t/3
    return np.array([v*np.cos(w),v*np.sin(w),w,v])

models = Net(8,32,4)
models.load_state_dict(torch.load('S.pkl'))
modeld = Net(8,32,4)
modeld.load_state_dict(torch.load('D.pkl'))
modelmd = Net(8,32,4)
modelmd.load_state_dict(torch.load('MD.pkl'))
modelms = Net(8,32,4)
modelms.load_state_dict(torch.load('MS.pkl'))
modeles = ControlNet(8,32,4)
modeles.load_state_dict(torch.load('ES_2000.pkl'))
def run_0(n,dt,case,seed):
    np.random.seed(seed)
    tau = 0.1
    nd = int(tau/dt)
    X = np.zeros([n+nd,4])
    DU = np.zeros([n,4])
    SU = np.zeros([n,4])
    # z = np.random.normal(0,1,n) # common noise
    z = np.random.normal(0,1,[n,4]) # uncorrelated noise
    for i in range(nd+1):
        t=i*dt-tau
        X[i,:] = initial(t)
    for i in range(n-1):
        x = X[i+nd,:]
        x_t = X[i,:]
        df = f(x)
        dg = g(x_t)
        if case == 0:
            X[i+nd+1,:] = x+df*dt+np.sqrt(dt)*z[i]*dg#+()*(dt*z[i]**2-dt)/(2*np.sqrt(dt))
        if case == 'S':
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((x,x_t))).to(torch.float32).unsqueeze(0)
                u = models(input).detach().numpy()
            X[i+nd+1,:]=x+df*dt+np.sqrt(dt)*z[i]*(dg+u)
            SU[i,:] = u
        if case == 'D':
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((x,x_t))).to(torch.float32).unsqueeze(0)
                u = modeld(input).detach().numpy()
            X[i+nd+1,:]=x+(df+u)*dt+np.sqrt(dt)*z[i]*(dg)
            DU[i,:] = u
        if case == 'M':
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((x,x_t))).to(torch.float32).unsqueeze(0)
                d_u = modelmd(input).detach().numpy()
                s_u = modelms(input).detach().numpy()
            X[i+nd+1,:]=x+(df+d_u)*dt+np.sqrt(dt)*z[i]*(dg+s_u)
            DU[i,:] = d_u
            SU[i,:] = s_u
        if case == 'NDC':
            with torch.no_grad():
                input = torch.from_numpy(np.concatenate((x,x_t))).to(torch.float32).unsqueeze(0)
                u = modeles(input).detach().numpy()*x
            X[i+nd+1,:]=x+(df+u)*dt+np.sqrt(dt)*z[i]*(dg)
            DU[i,:] = u
        if case == 'qp':
            u1,u2,u3,u4,d1 = qp(x,x_t)
            u = np.array([u1[0],u2[0],u3[0],u4[0]])
            X[i+nd+1,:]=x+(df+u)*dt+np.sqrt(dt)*z[i]*(dg)
            DU[i,:] = u
    return X,DU,SU

def energy(U,n=5000,dt=0.0001):
    a=np.linspace(0,dt*(n-1),n)
    e = 0.0
    for i in range(len(U)):
        e += integrate.trapz(np.array(np.sum(U[i,:]**2,axis=1)),a)
    return e/float(len(U))

def stop_time(X,delta=0.001,dt=0.0001):
    time = 0
    for i in range(len(X)):
        norm_x = np.sqrt(X[i,:,0]**2+X[i,:,1]**2)
        index = np.where(norm_x<delta)
        time += index[0][0]
    return time/float(len(X))*dt

def minima(X):
    min_x = 0
    for i in range(len(X)):
        norm_x = np.sqrt(X[i,:,0]**2+X[i,:,1]**2)
        min_x += np.min(norm_x)
        print(i,np.min(norm_x))
    return min_x/float(len(X))
'''
test
'''
seed = 3
n = 5000
dt = 0.0001
tau = 0.1
m = 10
font_size = 20

# X,DU,SU=run_0(n,dt,'S',seed)
# plt.plot(np.arange(len(X)),X[:,0])
# plt.plot(np.arange(len(X)),X[:,1])

# data = np.load('qp.npy',allow_pickle=True).item()
# X,DU,SU = data['X'],data['DU'],data['SU']
# X = np.delete(X,[2,4],axis=0)
# print(minima(X))
# plt.plot(np.arange(31000),np.mean(X[:,:,0],axis=0),color=colors[0],label=r'$x$')
# plt.plot(np.arange(31000),np.mean(X[:,:,1],axis=0),color=colors[1],label=r'$y$')
# plt.plot(np.arange(len(DU)),np.min(SU,axis=1),color=colors[2],label=r'$u$')
# a=np.linspace(0,dt*(len(DU)-1),len(DU))
# energy=integrate.trapz(np.array(np.sum(SU**2+DU**2,axis=1)),a)
# plt.title('{}'.format(energy))
'''
data generate
'''
# X,DU,SU = np.zeros([m,n+int(tau/dt),4]),np.zeros([m,n,4]),np.zeros([m,n,4])
# for i in range(m):
#     X[i,:],DU[i,:],SU[i,:] = run_0(n,dt,'qp',20*i)
#     print(i)
# np.save('qp_time.npy',{'X':X,'DU':DU,'SU':SU}) # (20000,0.0001)
# np.save('M.npy',{'X':X,'DU':DU,'SU':SU}) # throw out 2nd trajectory (50000,0.00001)

def subplot(X,xticks1,xticks2,yticks1,yticks2,ylim,title):
    alpha = 0.5
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


# data = np.load('ndc.npy',allow_pickle=True).item()
# X,DU,SU = data['X'],data['DU'],data['SU']
# X = np.delete(X,[2,4],axis=0)
# print(minima(X))
# print(stop_time(X,0.001))
# i = 0
# norm_x = np.sqrt(X[i,:,0]**2+X[i,:,1]**2)
# index = np.where(norm_x<1e-4)
# print(index[0][0],stop_time(X,1e-3))

# DU = np.delete(DU,[2,4],axis=0)
# SU = np.delete(SU,[2,4],axis=0)
# print(DU.shape,np.sum(DU[0,:]**2,axis=1).shape,energy(DU+SU,30000,0.0001))
# X = np.delete(X,3,axis=0)
# print(X.shape)
# for i in [5,6,7,8,9]:
#     plt.plot(np.arange(60000),X[i,:,0])
#     plt.plot(np.arange(60000),X[i,:,1])
# subplot(X,[0,1000,3500,6000],[-0.1,0,0.25,0.5],'uncontrol')

def plot():
    plt.subplot(231)
    data = np.load('uncontrol.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
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
    X = np.delete(X[:,0:60000:10,:],[2,4],axis=0)
    subplot(X,[0,1000,3500,6000],[-0.1,0,0.25,0.5],[0,1,2,3],[0,1,2,3],[-0.2,2.5],'NSC+M')
    plt.xlabel(r'$t$',fontsize=font_size)

# plot()

# plt.plot(X[:,0],X[:,1])
# plt.scatter(X[0,0],X[0,1],marker='*',s=300,color=colors[0]/max(colors[0]) * 0.7,zorder=10)
# plt.scatter(X[-1,0],X[-1,1],marker='o',s=300,color=colors[2]/max(colors[2]) * 0.7,zorder=10)
# plt.scatter(0.0,0.0,marker='v',s=300,color=colors[3]/max(colors[3]) * 0.7,zorder=10)
plt.show()