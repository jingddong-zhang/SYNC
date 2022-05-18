import torch
import torch.nn.functional as F
import numpy as np
import timeit
import argparse

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=float, default=2000)
parser.add_argument('--num', type=float, default=8)
parser.add_argument('--lr', type=float, default=0.05)
args = parser.parse_args()


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
        x = data[:,0:4]
        return out*x

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self._scontrol = ControlNet(n_input,n_hidden,n_output)
        self._dcontrol = ControlNet(n_input,n_hidden,n_output)

    def forward(self,data):
        s_u = self._scontrol(data)
        d_u = self._dcontrol(data)
        return d_u,s_u


def f_(data,u):
    x,y = data[:,0:4],data[:,4:8]
    z = torch.zeros_like(x)
    for i in range(len(x)):
        px,py,w,v=x[i,:]
        z[i,:] = torch.tensor([v*np.cos(w),v*np.sin(w),v,px**2+py**2])#+u[i]
    return z

def g_(data,u):
    x,y = data[:,0:4],data[:,4:8]
    z = torch.zeros_like(x)
    for i in range(len(x)):
        px,py,w,v=y[i,:]
        z[i,:] = torch.tensor([px,py,0.0,0.0])+u[i]
    return z


'''
For learning 
'''
N = args.N  # sample size
D_in = 8  # input dimension
H1 = 4 * D_in  # hidden dimension
D_out = 4  # output dimension
torch.manual_seed(10)
Data = torch.Tensor(N,8).uniform_(-10,10)
M , gamma = 4,5  #barrier

theta = 0.8
out_iters = 0
while out_iters < 1:
    # break
    start = timeit.default_timer()
    model = Net(D_in, H1, D_out)
    i = 0
    t = 0
    max_iters = 200
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while i < max_iters:
        d_u,s_u = model(Data)
        f = f_(Data,d_u)
        g = g_(Data,s_u)
        x = Data[:,0:4]
        v = x[:,3:4]
        # loss = (2-theta)*torch.diagonal(torch.mm(x, g.T))**2-torch.diagonal(torch.mm(x,x.T))*torch.diagonal(
        #     2*torch.mm(x,f.T)+torch.mm(g,g.T))
        loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
        # L_B = 2*(v-M/2)*f[:,3:4]/h(v)**2+g[:,3:4]**2/h(v)**2+4*g[:,3:4]**2*(v-M/2)**2/h(v)**3 - gamma*torch.log(1+torch.abs(h(v))) # barrier function 1
        # L_B = (2*(v-M/2)*f[:,3:4]/h(v)**2+g[:,3:4]**2/h(v)**2+4*g[:,3:4]**2*(v-M/2)**2/h(v)**3)
        # lossB = 2*L_B/h(v)-(1-theta)*(2*(v-M/2)*g[:,3:4])**2/h(v)**4
        AS_loss = (F.relu(-loss)).mean()
        print(i, "AS loss=", AS_loss.item())

        optimizer.zero_grad()
        AS_loss.backward()
        optimizer.step()

        # if AS_loss < 1e-8:
        #     break
        # if AS_loss<0.5:
        #     optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
        i += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)

    out_iters += 1
# torch.save(model._scontrol.state_dict(),'S_little.pkl')
# torch.save(model._dcontrol.state_dict(),'MD.pkl')
