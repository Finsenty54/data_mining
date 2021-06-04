import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy as sc
from scipy import random, linalg, stats, special

# 生成数据
NProperties = 1
NClasses = 4
NObjects = 200
symmetric_dirichlet = 1
distanceBTWclasses = 20
DiffBTWSpreadOFclasses = 2

# The mean vector
Mu = [np.random.random(NProperties)*distanceBTWclasses*i for i in range(1,NClasses+1)]
# the sd vector
Var = [np.random.random(NProperties)*DiffBTWSpreadOFclasses*i for i in range(1,NClasses+1)]

if symmetric_dirichlet==1:
    theta = np.repeat(1.0/NClasses,NClasses)
else:
    a = np.ones(NClasses)
    n = 1
    p = len(a)
    rd = np.random.gamma(np.repeat(a,n),n,p)
    rd = np.divide(rd,np.repeat(np.sum(rd),p))
    theta = rd

print 'The probabilities of each classes from 1 to '+str(NClasses)
print theta

r = np.random.multinomial(NObjects,theta)

# (3)
rAlln = [np.random.normal(Mu[i], Var[i], r[i]) for i in range(0,NClasses)]

# putting the generated data into an array form
y = rAlln[0]
for i in range(NClasses-1):
    y = np.hstack((y,rAlln[i+1]))

# Getting the true classes of the points 
v_true = np.zeros((1)) 
for i,j in enumerate(r):
    v_true = np.hstack((v_true, np.repeat(i+1, j)))

v_true = np.array(v_true[1:])
y_true = np.vstack((y, v_true))

# random shuffle the data points
np.random.shuffle(y_true.T)

y = y_true[0,:]

print 'The data:'
print y


v = np.array([random.randint(1, NClasses+1) for i in range(y.shape[0])])
print(v)

# E-step
def EStep(y, w, Mu, Sigma):
    
    # r_ij
    r_ij = np.zeros((y.shape[0], Mu.shape[0]))
    
    for Object in range(y.shape[0]):
        
        r_ij_Sumj = np.zeros(Mu.shape[0])
        
        # 计算不同分布下x的概率
        for jClass in range(Mu.shape[0]):
            r_ij_Sumj[jClass] = w[jClass] * sc.stats.norm.pdf(y[Object], Mu[jClass], np.sqrt(Sigma[jClass]))
        
        # 计算x
        for jClass in range(r_ij_Sumj.shape[0]):
            r_ij[Object,jClass] =   r_ij_Sumj[jClass] / np.sum(r_ij_Sumj)
        
    return r_ij