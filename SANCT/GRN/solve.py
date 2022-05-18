from scipy.optimize import fsolve
import numpy as np
from numpy import *
# 按格式要求定义我们需要求的函数
W = np.load('./data/topo_W.npy')

def f(x):
    return -x*1+np.matmul(x**2/(1+x**2),W.T)
    # return np.log(x) -np.log(1-x) + 2.2*(1-2*x)
# 调用fsolve函数

# sol_fsolve = fsolve(f, [-np.ones([1,100])*10]) # 第一个参数为我们需要求解的方程，第二个参数为方程解的估计值
# print(sol_fsolve)
# np.save('./data/solution',sol_fsolve)

