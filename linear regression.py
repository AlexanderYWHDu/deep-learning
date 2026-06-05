import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return(y_pred-y)**2

