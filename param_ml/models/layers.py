import torch 
from torch import nn

class LinearBn(nn.Module):
    def __init__(self, 
                 in_size, out_size, bias=True, do_batchnorm=True, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        if do_batchnorm:
            self.bn = nn.BatchNorm1d(out_size, eps=1e-5, momentum=0.1)
        else:
            self.bn = nn.Identity()
        self.activation = activation 

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x  