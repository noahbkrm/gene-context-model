import torch.nn as nn
from constants import HIDDEN_DIM

class Predictor(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim*4),
            nn.Linear(hidden_dim*4, hidden_dim)
        )
    
    def forward(self, z_context):
        z_context = self.net(z_context)
        return z_context