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

    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x].squeeze(1)


class Transformer(nn.Module):
    def __init__(self, d_in, d_hidden,n_heads):
        super().__init__()
        self.pos_embedding = PositionalEncoding(d_hidden)
        self.linear = nn.Linear(d_in, d_hidden)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_hidden, nhead=n_heads,dim_feedforward=d_hidden),num_layers=2)
        self.out = nn.Linear(d_hidden, d_in)
    def forward(self, x,t):
        x = self.linear(x)
        x = torch.cat((x,t),dim=1)
        x = self.transformer(x)
        x = x[:,0,:]
        x = self.out(x)
        return x