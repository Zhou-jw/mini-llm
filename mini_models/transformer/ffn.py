import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
        