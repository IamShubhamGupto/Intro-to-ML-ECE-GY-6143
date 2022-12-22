import numpy as np
def sigmoid(x, derivative = False):
    if derivative:
        return np.multiply(sigmoid(x),(1-sigmoid(x)))
    return(1/(1 + np.exp(-x))) 

def l2_loss(y, ypred, derivative=False):
    if derivative:
        return -1*np.sum(y-ypred)
    return np.sum((y-ypred)**2)
def relu(x):
    return max(0, x)

x1 = -9
Wh1x1 = 2.5
Wh1xb = 0.5
Wh2x1 = -2.5
Wh2xb = 3
zh_1 = Wh1x1*x1 + Wh1xb
uh1 = relu(zh_1)
zh_2 = Wh2x1*x1 + Wh2xb
uh2 = relu(zh_2)

Wo1h1 = 2
Wo1h2 = -0.5
Wo1hb = -1

zo_1 = Wo1h1*uh1 + Wo1h2*uh2 + Wo1hb
print(f"uo1 = {zo_1}")