import torch
import torch.nn.functional as F
import numpy as np
import timeit
import argparse

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=float, default=5000)
parser.add_argument('--num', type=float, default=3)
parser.add_argument('--lr', type=float, default=0.05)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out*x[:,:3]

def g_func(x):
    B,m0,m1=1,-1/7,-40/7
    return m0*x+(m1-m0)*(np.abs(x+B)-np.abs(x-B))/2

def f_(data):
    a,b,c,d=7,0.35,0.5,7
    x = data[:,0:3]
    z = torch.zeros_like(x)
    for i in range(len(x)):
        x1,x2,x3 = x[i,:]
        z[i,:] = torch.tensor([a*(x2-x1+40/7*torch.abs(x1)),b*(x1-x2)+c*x3,-d*x2])
    return z

def g_(data,u):
    x,y = data[:,0:3],data[:,3:6]
    alpha = 1.0
    beta = 1.0
    z = torch.zeros_like(x)
    for i in range(len(x)):
        sigma = alpha*torch.sum(torch.sin(2*x[i,:]))-beta*torch.sum(torch.sin(y[i,:]))
        z[i,:] = torch.ones(3)*sigma+u[i,:]
    return z


'''
For learning 
'''
N = args.N  # sample size
D_in = args.num*2  # input dimension
H1 = 4 * D_in  # hidden dimension
D_out = args.num # output dimension
torch.manual_seed(10)
Data = torch.Tensor(N,args.num*2).uniform_(-40,40)

theta = 0.8
out_iters = 0
while out_iters < 1:
    # break
    start = timeit.default_timer()

    model = Net(D_in, H1, D_out)

    i = 0
    t = 0
    max_iters = 500
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)

    while i < max_iters:
        out = model(Data)
        g = g_(Data,out)
        f = f_(Data)
        x = Data[:,0:3]
        # loss = (2 - theta) * torch.diagonal(torch.mm(x, g.T)) ** 2 - torch.diagonal(torch.mm(x, x.T)) * torch.diagonal(
        #     2 * torch.mm(x, f.T) + torch.mm(g, g.T))
        loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
        Lyapunov_risk = (F.relu(-loss)).mean()
        # Lyapunov_risk.requires_grad_(True)

        print(i, "Lyapunov Risk=", Lyapunov_risk.item())

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step()

        if Lyapunov_risk < 1e-8:
            break
        i += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)

    out_iters += 1
    torch.save(model.state_dict(),'AS_chua.pkl')