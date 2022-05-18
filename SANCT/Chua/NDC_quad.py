import torch
import torch.nn.functional as F
import torch.nn as nn
import timeit
import matplotlib.pyplot as plt
from hessian import hessian
from hessian import jacobian
import numpy as np
import argparse
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--D_in', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--niters', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1000)
args = parser.parse_args()

torch.manual_seed(10)

class ControlNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ControlNet, self).__init__()
        # torch.manual_seed(2)
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_output)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        return self.net(x)

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
        self._control=ControlNet(6,24,3)
        self._w = ControlNet(3,6,1)
        self._gamma = GammaNet(1,6,1)

    def forward(self,data):
        sigmoid=torch.nn.Tanh()
        t,x,y= data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
        t_x = data[:,0:args.D_in+1]
        h_1=sigmoid(self.layer1(x))
        out=self.layer2(h_1)**2
        u = self._control(torch.cat((x,y),1))*x
        w1 = self._w(x)**2 + 0.0001*torch.exp(-1/torch.sum(x**2))
        w2 = self._w(y)**2
        gamma = self._gamma(t)
        return out,u,w1,w2,gamma


def g_func(x):
    B,m0,m1=1,-1/7,-40/7
    return m0*x+(m1-m0)*(np.abs(x+B)-np.abs(x-B))/2


def f_(data,u):
    a,b,c,d=7,0.35,0.5,7
    # x=data[:,0:3]
    t,x,y=data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
    z=torch.zeros_like(x)
    for i in range(len(x)):
        x1,x2,x3=x[i,:]
        z[i,:]=torch.tensor([a*(x2-x1+40/7*torch.abs(x1)),b*(x1-x2)+c*x3,-d*x2])+u[i,:]
    return z


def g_(data,u):
    # x,y=data[:,0:3],data[:,3:6]
    t,x,y=data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
    alpha=1.0
    beta=1.0
    z=torch.zeros_like(x)
    for i in range(len(x)):
        sigma=alpha*torch.sum(torch.sin(2*x[i,:]))-beta*torch.sum(torch.sin(y[i,:]))
        z[i,:]=torch.ones(3)*sigma
    return z

def get_batch(data):
    s = torch.from_numpy(np.random.choice(np.arange(args.N, dtype=np.int64), args.batch_size, replace=False))
    batch_x = data[s,:]  # (M, D)
    return batch_x


'''
For learning 
'''
N = args.N  # sample size
D_in = 3  # input dimension
H1 = 4*D_in  # hidden dimension
D_out = 1  # output dimension

xy_data = torch.Tensor(N,6).uniform_(-50,50)
t_data = torch.Tensor(N,1).uniform_(0,10)
Data = torch.cat((t_data,xy_data),1)
l=0.0001
# f=f(data)
# print(f.shape)
start=timeit.default_timer()
model = Net(D_in,H1,D_out)
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
# optimizer=torch.optim.Adam([i for i in model.parameters()]+[model.k],lr=args.lr)
max_iters=200
for k in range(1,args.niters+1):
# for k in range(1):
    # break
    data = get_batch(Data)
    Loss = []
    i=0
    while i<max_iters:
        # break
        V_net,u,w1,w2,gamma=model(data)
        W1=model.layer1.weight
        W2=model.layer2.weight
        B1=model.layer1.bias
        B2=model.layer2.bias

        f = f_(data,u)
        g = g_(data,u)
        # t,x,y=data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
        x = data[:,1:args.D_in+1].clone().detach().requires_grad_(True)
        # x,y= data[:,0:args.D_in],data[:,args.D_in:2*args.D_in]
        output = torch.mm(torch.tanh(torch.mm(x,W1.T)+B1),W2.T)+B2

        V=torch.sum(output**2)
        Vx=jacobian(V,x)
        Vxx=hessian(V,x)
        loss=torch.zeros(args.batch_size)
        for r in range(args.batch_size):
            # L_V = torch.sum(2*l*x[r,:]*f[r,:])+torch.sum(Vx[0,D_in*r+1:D_in*r+D_in]*f[r,:])+0.5*torch.mm(g[r,:].unsqueeze(0),
            #         torch.mm(Vxx[D_in*r+1:D_in*r+D_in,D_in*r+1:D_in*r+D_in],g[r,:].unsqueeze(1)))+0.5*torch.sum(2*l*g[r,:]**2)
            L_V=torch.sum(2*l*x[r,:]*f[r,:])+torch.sum(Vx[0,D_in*r:D_in*r+D_in]*f[r,:])+0.5*torch.mm(
                g[r,:].unsqueeze(0),torch.mm(Vxx[D_in*r:D_in*r+D_in,D_in*r:D_in*r+D_in],g[r,:].unsqueeze(1)))+0.5*torch.sum(
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
        if Lasalle_risk<0.001:
            break

        # stop = timeit.default_timer()
        # print('per:',stop-start)

        i+=1
    # print(q)

stop=timeit.default_timer()
print('\n')
print("Total time: ",stop-start)
# print(model.k)
torch.save(model._control.state_dict(),'ES_2.pkl')