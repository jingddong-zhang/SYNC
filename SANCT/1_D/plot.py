import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass

font_size = 25
X = torch.load('loss.pt').detach().numpy() # load loss data
plt.plot(np.arange(len(X)),X)
plot_grid()
plt.xlabel('Training Epoch',fontsize=font_size)
plt.ylabel('LaSalle Loss',fontsize=font_size)
plt.show()

