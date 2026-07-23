import torch.nn as nn
from constants import HIDDEN_DIM

class Projection(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim*4),
            nn.Linear(hidden_dim*4, hidden_dim//3)
        )
    
    def forward(self, z):
        z = self.net(z)
        return z