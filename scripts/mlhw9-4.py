import numpy as np

def sigmoid(x, derivative = False):
    if derivative:
        return np.multiply(sigmoid(x),(1-sigmoid(x)))
    return(1/(1 + np.exp(-x))) 

def l2_loss(y, ypred, derivative=False):
    if derivative:
        return -1*np.sum(y-ypred)
    return np.sum((y-ypred)**2)

x1 = 0
x2 = 1
Wh1x1 = 1.5
Wh2x1 = -2.5
h1xb = 2.5
h2xb = 1
Wh1x2 = 1
Wh2x2 = -1
zh1 = x1*Wh1x1 + x2*Wh1x2 + h1xb
uh1 = sigmoid(zh1)
print(f"uh_1 = {uh1}")
zh2 = x1*Wh2x1 + x2*Wh2x2 + h2xb
uh2 = sigmoid(zh2)
print(f"uh_2 = {uh2}")
Wo1uh1 = 1
Wo1uh2 = 1
hb = -2
zo1 = uh1*Wo1uh1 + uh2*Wo1uh2 + hb
uo1 = sigmoid(zo1)
print(f"uo_1 = {uo1}")

y = 1
sigma_o = l2_loss(y, uo1, derivative=True)*sigmoid(zo1, derivative=True)
print(f"sigma_o = {sigma_o}")
sigma_h1 = sigma_o*Wo1uh1*sigmoid(zh1, derivative=True)
print(f"sigma_h1 = {sigma_h1}")
sigma_h2 = sigma_o*Wo1uh2*sigmoid(zh2, derivative=True)
print(f"sigma_h2 = {sigma_h2}")

sigma_wo1 = np.array([sigma_o*uh1, sigma_o*uh2, sigma_o])
print(f"sigma_wo1 = {sigma_wo1}")
sigma_wh1 = np.array([sigma_h1*x1, sigma_h1*x2, sigma_h1])
print(f"sigma_wh1 = {sigma_wh1}")
sigma_wh2 = np.array([sigma_h2*x1, sigma_h2*x2, sigma_h2])
print(f"sigma_wh2 = {sigma_wh2}")

alpha = 0.3
Wo1 = np.array([Wo1uh1, Wo1uh2, hb]) - alpha*sigma_wo1
print(f"Wo1 = {Wo1}")
Wh1 = np.array([Wh1x1, Wh1x2, h1xb]) - alpha*sigma_wh1
print(f"Wh1 = {Wh1}")
Wh2 = np.array([Wh2x1, Wh2x2, h2xb]) - alpha*sigma_wh2
print(f"Wh2 = {Wh2}")


