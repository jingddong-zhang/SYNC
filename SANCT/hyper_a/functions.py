import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import timeit 
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
import numpy as np

np.random.seed(10)


class ControlNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ControlNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, data):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(data))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        x = data[:,0:2]
        return out*x

class ICNN(nn.Module):
    def __init__(self, input_shape, layer_sizes, activation_fn):
        super(ICNN, self).__init__()
        self._input_shape = input_shape
        self._layer_sizes = layer_sizes
        self._activation_fn = activation_fn
        ws = []
        bs = []
        us = []
        prev_layer = input_shape
        w = torch.empty(layer_sizes[0], *input_shape)
        nn.init.xavier_normal_(w)
        ws.append(nn.Parameter(w))
        b = torch.empty([layer_sizes[0], 1])
        nn.init.xavier_normal_(b)
        bs.append(nn.Parameter(b))
        for i in range(len(layer_sizes))[1:]:
            w = torch.empty(layer_sizes[i], *input_shape)
            nn.init.xavier_normal_(w)
            ws.append(nn.Parameter(w))
            b = torch.empty([layer_sizes[i], 1])
            nn.init.xavier_normal_(b)
            bs.append(nn.Parameter(b))
            u = torch.empty([layer_sizes[i], layer_sizes[i-1]])
            nn.init.xavier_normal_(u)
            us.append(nn.Parameter(u))
        self._ws = nn.ParameterList(ws)
        self._bs = nn.ParameterList(bs)
        self._us = nn.ParameterList(us)

    def forward(self, x):
        # x: [batch, data]
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        else:
            data_dims = list(range(1, len(self._input_shape) + 1))
            x = x.permute(*data_dims, 0)
        z = self._activation_fn(torch.addmm(self._bs[0], self._ws[0], x))
        for i in range(len(self._us)):
            u = F.softplus(self._us[i])
            w = self._ws[i + 1]
            b = self._bs[i + 1]
            z = self._activation_fn(torch.addmm(b, w, x) + torch.mm(u, z))
        return z

def h(x):
    M = 6
    # return M**2-torch.sum(x**2,dim=1)
    return M**2-torch.sum(x[:,0:1]**2,dim=1)

class KFunction(nn.Module):
    def __init__(self,input_shape,smooth_relu_thresh=0.1,layer_sizes=[64,64],lr=3e-4):
        super(KFunction,self).__init__()
        torch.manual_seed(2)
        self._d=smooth_relu_thresh
        self._icnn=ICNN(input_shape,layer_sizes,self.smooth_relu)
        self._scontrol = ControlNet(4,4*4,2)
        self._dcontrol = ControlNet(4,4*4,2)

    def forward(self,data):
        x = data[:,0:2]
        g=self._icnn(h(x))
        g0=self._icnn(torch.zeros_like(h(x)))
        s_u = self._scontrol(data)
        d_u = self._dcontrol(data)
        return 1/self.smooth_relu(g-g0),d_u,s_u

    def smooth_relu(self,x):
        relu=x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        sq=(2*self._d*relu.pow(3)-relu.pow(4))/(2*self._d**3)
        lin=x-self._d/2
        return torch.where(relu<self._d,sq,lin)


def lya(ws,bs,us,smooth,x,input_shape):
    if len(x.shape)<2:
        x=x.unsqueeze(0)
    else:
        data_dims=list(range(1,len(input_shape)+1))
        x=x.permute(*data_dims,0)
    z=smooth(torch.addmm(bs[0],ws[0],x))
    for i in range(len(us)):
        u=F.softplus(us[i])
        w=ws[i+1]
        b=bs[i+1]
        z=smooth(torch.addmm(b,w,x)+torch.mm(u,z))
    return z


def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.3, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.3, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass

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
# ang = torch.zeros([5,1]) #initial angle
# vel = torch.zeros([5,1]) #initial velocity
# for i in range(5):
#     x0 = np.random.uniform(-6,6,2)
#     ang[i,0] = x0[0]
#     vel[i,0] = x0[1]

def invert_pendulum(state0, t):
    state0 = state0.flatten()
    G = 9.81  # gravity
    L = 0.5   # length of the pole 
    m = 0.15  # ball mass
    b = 0.1   # friction
    def f(state,t):
        x, y = state  # unpack the state vector
        return y, G*np.sin(x)/L +(-b*y)/(m*L**2) # derivatives
    states = odeint(f, state0, t)
    return states.transpose()

def fluid(y) :
    #parameters
    G = 9.81
    L = 0.5
    m = 0.15
    b = 0.1
    x1,x2 = y
    dydt =[x2,  (m*G*L*np.sin(x1) - b*x2) / (m*L**2)]
    return dydt

#绘制向量场
def Plotflow(Xd, Yd):
    # Plot phase plane 
    DX, DY = fluid([Xd, Yd])
    DX=DX/np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY=DY/np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.6, arrowstyle='-|>', arrowsize=1.5)

