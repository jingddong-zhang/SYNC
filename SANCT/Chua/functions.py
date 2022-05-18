import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


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

class ICNN(nn.Module):
    def __init__(self,input_shape,layer_sizes,activation_fn):
        super(ICNN,self).__init__()
        self._input_shape=input_shape
        self._layer_sizes=layer_sizes
        self._activation_fn=activation_fn
        ws=[]
        bs=[]
        us=[]
        prev_layer=input_shape
        w=torch.empty(layer_sizes[0],*input_shape)
        nn.init.xavier_normal_(w)
        ws.append(nn.Parameter(w))
        b=torch.empty([layer_sizes[0],1])
        nn.init.xavier_normal_(b)
        bs.append(nn.Parameter(b))
        for i in range(len(layer_sizes))[1:]:
            w=torch.empty(layer_sizes[i],*input_shape)
            nn.init.xavier_normal_(w)
            ws.append(nn.Parameter(w))
            b=torch.empty([layer_sizes[i],1])
            nn.init.xavier_normal_(b)
            bs.append(nn.Parameter(b))
            u=torch.empty([layer_sizes[i],layer_sizes[i-1]])
            nn.init.xavier_normal_(u)
            us.append(nn.Parameter(u))
        self._ws=nn.ParameterList(ws)
        self._bs=nn.ParameterList(bs)
        self._us=nn.ParameterList(us)

    def forward(self,x):
        # x: [batch, data]
        if len(x.shape)<2:
            x=x.unsqueeze(0)
        else:
            data_dims=list(range(1,len(self._input_shape)+1))
            x=x.permute(*data_dims,0)
        z=self._activation_fn(torch.addmm(self._bs[0],self._ws[0],x))
        for i in range(len(self._us)):
            u=F.softplus(self._us[i])
            w=self._ws[i+1]
            b=self._bs[i+1]
            z=self._activation_fn(torch.addmm(b,w,x)+torch.mm(u,z))
        return z


# class ControlNet(torch.nn.Module):
#     def __init__(self, n_input, n_hidden, n_output):
#         super(ControlNet, self).__init__()
#         torch.manual_seed(2)
#         self.layer1 = torch.nn.Linear(n_input, n_hidden)
#         self.layer2 = torch.nn.Linear(n_hidden,n_hidden)
#         self.layer3 = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         sigmoid = torch.nn.ReLU()
#         h_1 = sigmoid(self.layer1(x))
#         h_2 = sigmoid(self.layer2(h_1))
#         out = self.layer3(h_2)
#         return out

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

class LyapunovFunction(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,input_shape,smooth_relu_thresh=0.1,layer_sizes=[64,64],lr=3e-4,
                 eps=1e-3):
        super(LyapunovFunction,self).__init__()
        torch.manual_seed(2)
        self._d=smooth_relu_thresh
        self._icnn=ICNN(input_shape,layer_sizes,self.smooth_relu)
        self._eps=eps
        self._control=ControlNet(n_input,n_hidden,n_output)

    def forward(self,x):
        g=self._icnn(x)
        g0=self._icnn(torch.zeros_like(x))
        u=self._control(x)
        u0=self._control(torch.zeros_like(x))
        return self.smooth_relu(g-g0)+self._eps*x.pow(2).sum(dim=1),u*x
        # return self.smooth_relu(g - g0) + self._eps * x.pow(2).sum(dim=1), u-u0

    def smooth_relu(self,x):
        relu=x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        sq=(2*self._d*relu.pow(3)-relu.pow(4))/(2*self._d**3)
        lin=x-self._d/2
        return torch.where(relu<self._d,sq,lin)

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass

