import torch
from torch import nn

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # (batch_size, seq_len, 1) 
        std = x.std(-1, keepdim=True)        
        norm = self.weight * (x - mean) / std + self.eps + self.bias # (batch_size, seq_len, d_model)
        return norm
