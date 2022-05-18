import torch
import torch.nn.functional as F
import torch.nn as nn
import timeit
import matplotlib.pyplot as plt
from hessian import hessian
from hessian import jacobian
import numpy as np
import math
import argparse
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--D_in', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--niters', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

torch.manual_seed(10)

# class ControlNet(torch.nn.Module):
#     def __init__(self, n_input, n_hidden, n_output):
#         super(ControlNet, self).__init__()
#         # torch.manual_seed(2)
#         self.net = nn.Sequential(
#             nn.Linear(n_input, n_hidden),
#             nn.ReLU(),
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(),
#             nn.Linear(n_hidden,n_output)
#         )
#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.001)
#                 nn.init.constant_(m.bias, val=0)
#
#     def forward(self, x):
#         return self.net(x)

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
        return out

class GammaNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(GammaNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, t):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(t))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)**2
        return out*torch.exp(-t)

class Net(torch.nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        torch.manual_seed(2)
        self.layer1=torch.nn.Linear(n_input,n_hidden)
        self.layer2=torch.nn.Linear(n_hidden,n_output)
        self._control=ControlNet(200,4*200,100)
        self._w = ControlNet(100,200,1)
        self._gamma = GammaNet(1,6,1)

    def forward(self,data):
        sigmoid=torch.nn.Tanh()
        x,y= data[:,0:100],data[:,100:200]
        h_1=sigmoid(self.layer1(x))
        out=self.layer2(h_1)**2
        u = self._control(data)*x/50
        w1 = self._w(x)**2 + 0.0001*torch.exp(-1/torch.sum(x**2))
        w2 = self._w(y)**2
        # gamma = self._gamma(t)
        return out,u,w1,w2 #,gamma

W = np.load('./data/topo_W.npy',allow_pickle=True)
sol = np.load('./data/solution.npy')
W = torch.from_numpy(W).to(torch.float32)
sol = torch.from_numpy(sol).to(torch.float32)
def f_(data,u):
    x,y = data[:,0:100],data[:,100:200]
    z = torch.zeros_like(x)
    for i in range(len(x)):
        s = x[i,:]
        z[i,:] = -s+torch.mm((s**2/(1+s**2)).unsqueeze(0),W.T)+u[i,:]
    return z

def g_(data,u):
    x,y = data[:,0:100],data[:,100:200]
    z = torch.zeros_like(x)
    for i in range(len(x)):
        s = y[i,:]
        z[i,:] = torch.sin(s/sol*math.pi)#+u[i]
    return z

def get_batch(data):
    s = torch.from_numpy(np.random.choice(np.arange(args.N, dtype=np.int64), args.batch_size, replace=False))
    batch_x = data[s,:]  # (M, D)
    return batch_x


'''
For learning 
'''
N = args.N  # sample size
D_in = 100  # input dimension
H1 = 4*D_in  # hidden dimension
D_out = 1  # output dimension
Data = torch.Tensor(N,200).uniform_(-5,5)
l=0.0001
start=timeit.default_timer()
model = Net(D_in,H1,D_out)
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
# optimizer=torch.optim.Adam([i for i in model.parameters()]+[model.k],lr=args.lr)
max_iters=30
for k in range(1,args.niters+1):
# for k in range(1):
    # break
    data = get_batch(Data)
    Loss = []
    i=0
    while i<max_iters:
        # break
        V_net,u,w1,w2=model(data)
        W1=model.layer1.weight
        W2=model.layer2.weight
        B1=model.layer1.bias
        B2=model.layer2.bias

        f = f_(data,u)
        g = g_(data,u)
        x = data[:,0:100].clone().detach().requires_grad_(True)
        # x,y= data[:,0:args.D_in],data[:,args.D_in:2*args.D_in]
        # output = torch.mm(torch.tanh(torch.mm(x,W1.T)+B1),W2.T)+B2
        # V=torch.sum(output**2)
        # Vx=jacobian(V,x)
        # print(Vx.shape)
        # Vxx=hessian(V,x)
        # print(Vxx.shape)
        def V_func(x):
            output=torch.mm(torch.tanh(torch.mm(x,W1.T)+B1),W2.T)+B2
            return torch.sum(output**2)
        Vx =  torch.autograd.functional.jacobian(V_func,x)
        Vxx = torch.autograd.functional.hessian(V_func,x)
        # print(Vx.shape,Vxx.shape)
        loss=torch.zeros(args.batch_size)
        for r in range(args.batch_size):
            # L_V = torch.sum(2*l*x[r,:]*f[r,:])+torch.sum(Vx[0,D_in*r+1:D_in*r+D_in]*f[r,:])+0.5*torch.mm(g[r,:].unsqueeze(0),
            #         torch.mm(Vxx[D_in*r+1:D_in*r+D_in,D_in*r+1:D_in*r+D_in],g[r,:].unsqueeze(1)))+0.5*torch.sum(2*l*g[r,:]**2)
            L_V=torch.sum(2*l*x[r,:]*f[r,:])+torch.sum(Vx[r,:]*f[r,:])+0.5*torch.mm(
                g[r,:].unsqueeze(0),torch.mm(Vxx[r,:,r,:],g[r,:].unsqueeze(1)))+0.5*torch.sum(
                2*l*g[r,:]**2)
            # V_t = Vx[0,D_in*r]
            # loss[r]=L_V+V_t+w1[r]-w2[r]-gamma[r]
            # print(L_V.shape,w1[r].shape)
            loss[r]=L_V+w1[r]-w2[r]

        Lasalle_risk=(F.relu(loss)).mean()
        # Loss.append(Lyapunov_risk)
        print(k,i,"Lyapunov Risk=",Lasalle_risk.item())

        optimizer.zero_grad()
        Lasalle_risk.backward()
        optimizer.step()
        if Lasalle_risk<0.0005:
            break

        # stop = timeit.default_timer()
        # print('per:',stop-start)

        i+=1
    # print(q)

stop=timeit.default_timer()
print('\n')
print("Total time: ",stop-start)
torch.save(model._control.state_dict(),'./data/ES.pkl')