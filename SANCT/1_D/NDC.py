import torch
import torch.nn.functional as F
import timeit
import matplotlib.pyplot as plt
from hessian import hessian
from hessian import jacobian
import numpy as np
import argparse
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=int, default=500)
parser.add_argument('--D_in', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--niters', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=500)
args = parser.parse_args()



class ControlNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ControlNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
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
        self._control=ControlNet(2,4,1)
        # self.k = torch.tensor([[0.1,0.2]],requires_grad=True)
        self._w1 = ControlNet(1,6,1)
        self._gamma = GammaNet(1,6,1)

    def forward(self,data):
        sigmoid=torch.nn.Tanh()
        t,x,y= data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
        t_x = data[:,0:args.D_in+1]
        h_1=sigmoid(self.layer1(x))
        out=self.layer2(h_1)**2
        u=self._control(torch.cat((x,y),1))*x
        # u = torch.sum(self.k*data,dim=1)
        w1 = self._w1(x)**2 + 0.0001*torch.exp(-1/x**2)
        w2 = self._w1(y)**2
        gamma = self._gamma(t)
        return out,u,w1,w2,gamma


def f_(data,u):
    t,x,y= data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
    z = torch.zeros_like(x)
    a1 = 1
    b1 = 1
    c1 = 100
    for i in range(len(x)):
        z[i] = a1*x[i]+b1*y[i]
    return z

def g_(data,u):
    t,x,y= data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
    z = torch.zeros_like(x)
    a2 = 0
    b2 = 1
    for i in range(len(x)):
        z[i] = a2*x[i]+b2*y[i]+u[i]
    return z

def get_batch(data):
    s = torch.from_numpy(np.random.choice(np.arange(args.N, dtype=np.int64), args.batch_size, replace=False))
    batch_x = data[s,:]  # (M, D)
    return batch_x


'''
For learning 
'''
N = args.N  # sample size
D_in = 1  # input dimension
H1 = 24*D_in  # hidden dimension
D_out = 1  # output dimension
torch.manual_seed(10)
xy_data = torch.Tensor(N,2).uniform_(-5,5)
t_data = torch.Tensor(N,1).uniform_(0,10)
Data = torch.cat((t_data,xy_data),1)
l=0.0001
# f=f(data)
# print(f.shape)
start=timeit.default_timer()
model = Net(D_in,H1,D_out)
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
# optimizer=torch.optim.Adam([i for i in model.parameters()]+[model.k],lr=args.lr)
max_iters=1000
for k in range(1,args.niters+1):
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
        t,x,y=data[:,0:1],data[:,1:args.D_in+1],data[:,args.D_in+1:2*args.D_in+1]
        x = data[:,1:args.D_in+1].clone().detach().requires_grad_(True)
        output = torch.mm(torch.tanh(torch.mm(x,W1.T)+B1),W2.T)+B2

        V=torch.sum(output**2)
        Vx=jacobian(V,x)
        Vxx=hessian(V,x)
        loss=torch.zeros(args.batch_size)

        for r in range(args.batch_size):
            L_V = torch.sum(2*l*x[r,:]*f[r,:])+torch.sum(Vx[0,D_in*r:D_in*r+D_in]*f[r,:])+0.5*torch.mm(g[r,:].unsqueeze(0),
                    torch.mm(Vxx[D_in*r:D_in*r+D_in,D_in*r:D_in*r+D_in],g[r,:].unsqueeze(1)))+0.5*torch.sum(2*l*g[r,:]**2)
            loss[r]=L_V+w1[r]-w2[r]

        Lasalle_risk=(F.relu(loss)).mean()
        Loss.append(Lasalle_risk)
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
torch.save(torch.tensor(Loss),'loss.pt')