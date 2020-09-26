# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import packages
import numpy as np
import torch, torchvision
from matplotlib import pyplot as plt


#%% OLS by PyTorch autograd
# simulate design matrix 
n = 200
p = 5
x = torch.randn([n,p])
x[:,0] = 1.
bet = torch.tensor([-2.,-1.,0.,1.,2.]) # true regression coefficients
y = torch.matmul(x,bet) + torch.randn(x.size(0))

# OLS estimation
bet_OLS = torch.matmul(torch.matmul(x.t(),x).inverse(), torch.matmul(x.t(),y))
print(bet_OLS)

# OLS by gradient descent
bet_gd = torch.zeros([p], requires_grad=True)
print(bet_gd)


losses = []
eta = 0.01 # step size
for _ in range(300):
    yhat = torch.matmul(x,bet_gd)
    mse_loss = (y-yhat).pow(2).mean()
    mse_loss.backward()
    with torch.no_grad(): # temporarily set all the requires_grad flag to false
    # We don't want PyTorch to calculate the gradients of the new defined 
    # variables and just want to update their values.
    # Alternatively, one can use .detach(); read the full tutorial: 
    # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients
        bet_gd -= bet_gd.grad * eta
        bet_gd.grad.zero_()
    losses += [mse_loss.item()]

print('Ground truth: ', bet.numpy())
print('Gradient descent: ', bet_gd.detach().numpy())
plt.plot(losses)