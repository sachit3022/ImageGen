import torch
import numpy as np
import math
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.BatchNorm1d(d_out),
            nn.Dropout(dropout)
        )
        self.net.apply(weights_init)
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_blocks, dropout=0.1):
        super().__init__()
        self.in_block = MLPBlock(d_in, d_hidden, dropout)
        self.hidden_blocks = nn.ModuleList([MLPBlock(d_hidden, d_hidden, dropout) for _ in range(n_blocks)])
        self.out_block = nn.Linear(d_hidden, d_out)
    
    def forward(self, x):
        x = self.in_block(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = self.out_block(x)
        return x

    