import torch
import torch.nn.functional as F
import numpy as np
import timeit
import argparse
from hessian import hessian
from hessian import jacobian
from functions import *
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=float, default=1000)
parser.add_argument('--num', type=float, default=4)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--niters', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1000)
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)

setup_seed(10)

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


def get_batch(data):
    s=torch.from_numpy(np.random.choice(np.arange(args.N,dtype=np.int64),args.batch_size,replace=False))
    batch_x=data[s,:]  # (M, D)
    return batch_x

def f_(data,u):
    x,y = data[:,0:2],data[:,2:4]
    z = torch.zeros_like(x)
    G=9.81  # gravity
    L=0.5  # length of the pole
    m=0.15  # ball mass
    b=0.1  # friction
    for i in range(len(x)):
        w,v = x[i,:]
        z[i,:] = torch.tensor([v, G*np.sin(w)/L +(-b*v)/(m*L**2)])#+u[i]
    return z

def g_(data,u):
    x,y = data[:,0:2],data[:,2:4]
    z = torch.zeros_like(x)
    for i in range(len(x)):
        w,v=y[i,:]
        z[i,:] = torch.tensor([torch.sin(w),torch.sin(v)])+u[i]
    return z


'''
For learning 
'''

N = args.N  # sample size
D_in = 4  # input dimension
H1 = 4 * D_in  # hidden dimension
D_out = 2  # output dimension
torch.manual_seed(10)
Data = torch.Tensor(N,4).uniform_(-5,5)
T_M = torch.kron(torch.eye(args.batch_size),torch.ones([2,1]))
theta = 0.8
gamma = 5.0
max_iters = 200


start = timeit.default_timer()
for k in range(1,args.niters+1):
    data = get_batch(Data)
    out_iters=0
    while out_iters < 1:
        # break

        model = KFunction((1,),0.01,[6,6,1])
        i = 0
        t = 0

        learning_rate = args.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        while i < max_iters:
            B,d_u,s_u = model(data)
            f = f_(data,d_u)
            g = g_(data,s_u)
            x = data[:,0:2]
            w = data[:,0:1]
            # loss = (2-theta)*torch.diagonal(torch.mm(x, g.T))**2-torch.diagonal(torch.mm(x,x.T))*torch.diagonal(
            #     2*torch.mm(x,f.T)+torch.mm(g,g.T))
            loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
            '''
            barrier loss with prior
            '''
            # L_B = 2*(v-M/2)*f[:,3:4]/h(v)**2+g[:,3:4]**2/h(v)**2+4*g[:,3:4]**2*(v-M/2)**2/h(v)**3 - gamma*torch.log(1+torch.abs(h(v))) # barrier function 1
            # L_B = 2*torch.sum(w*f[:,0:1],dim=1)/h(w)**2+torch.sum(g[:,0:1]**2,dim=1)/h(x)**2+4*torch.sum((w*g[:,0:1])**2,dim=1)/h(x)*3
            # lossB = 2*L_B/h(w)-(1-theta)*(2*torch.sum(w*g[:,0:1],dim=1)/h(w)**2)**2
            # lossB = L_B-gamma*torch.log(1+h(x))
            '''
            barrier loss with NN
            '''
            x=x.clone().detach().requires_grad_(True)
            ws=model._icnn._ws
            bs=model._icnn._bs
            us=model._icnn._us
            smooth=model.smooth_relu
            input_shape=(D_in,)
            V1=lya(ws,bs,us,smooth,h(x),input_shape)
            V0=lya(ws,bs,us,smooth,torch.zeros_like(h(x)),input_shape)
            num_V=1/smooth(V1-V0)
            # V=torch.sum(1/smooth(V1-V0))
            # Vx=jacobian(V,x)
            # Vxx=hessian(V,x)
            # print(Vx.shape,Vxx.shape)
            def V_func(x):
                V1=lya(ws,bs,us,smooth,h(x),input_shape)
                V0=lya(ws,bs,us,smooth,torch.zeros_like(h(x)),input_shape)
                V=torch.sum(1/smooth(V1-V0))
                return V
            Vx=torch.autograd.functional.jacobian(V_func,x).view([1,2*args.batch_size])
            Vxx = torch.autograd.functional.hessian(V_func,x).view([2*args.batch_size,2*args.batch_size])
            # print(Vxx.shape)
            g_v=g.view([1,args.batch_size*2])
            L_V=torch.sum(f*Vx.view([args.batch_size,2]),dim=1).unsqueeze(0)+torch.mm(
                torch.diag(torch.mm(Vxx,torch.mm(g_v.T,g_v))).unsqueeze(0),T_M)/2
            Vxg=torch.sum(g*Vx.view([args.batch_size,2]),dim=1).unsqueeze(0)
            lossB = 2*num_V*L_V-(1-theta)*Vxg**2
            # lossB=L_V-gamma*torch.log(1+h(x))
            AS_loss = (F.relu(-loss)).mean()+(F.relu(lossB)).mean()
            # AS_loss = (F.relu(lossB)).mean()



            print(k,i, "AS loss=", AS_loss.item())

            optimizer.zero_grad()
            AS_loss.backward()
            optimizer.step()
            if AS_loss < 2e-4:
                optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
            if AS_loss < 1e-5:
                break
            i += 1


        # print('\n')
        # print("Total time: ", stop - start)
        # print("Verified time: ", t)

        out_iters += 1

    stop=timeit.default_timer()
    print('\n')
    print("Total time: ",stop-start)
torch.save(model._scontrol.state_dict(),'safe_S_1000.pkl')
# torch.save(model._scontrol.state_dict(),'safe_S_2000.pkl')
# torch.save(model._scontrol.state_dict(),'safe2_S.pkl')
