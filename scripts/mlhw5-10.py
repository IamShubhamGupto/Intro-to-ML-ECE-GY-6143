from numpy import log as ln
from math import e
import numpy as np
def fun(x,neg=False):
    if neg:
        return ((e**-x)/(1+e**-x))
    return (1/(1+e**-x))
x = np.array([[1,2,2],
    [1,-3,2],
    [1,-1,0],
    [1,-2,2]])
y = np.array([0,1,0,1])
w = np.array([1.,-4.,-3.])
z = np.array(np.matmul(x,w))
print("z = ",z)
px = np.array([fun(z_i, False) for z_i in z])
print(px)
loss = -1*np.sum((y*ln(1/(1+e**-z)) + (1-y)*ln(e**-z/(1+e**-z))))
print("loss = ",loss)
alpha = 0.6
print("this ", np.sum(x*((y-(1/(1+e**-z))))[:,None], axis=0))
nw = w+ alpha*np.sum(x*((y-(1/(1+e**-z))))[:,None], axis=0)
print("new weights = ", nw)
nz = np.array(np.matmul(x,nw))
print("new z", nz)
nloss = -1*np.sum((y*ln(1/(1+e**-nz)) + (1-y)*ln(e**-nz/(1+e**-nz))))
print("new loss",nloss)


