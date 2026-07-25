import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from constants import HIDDEN_DIM

class Projection(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        out_dim = 256
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, out_dim)
        )
        self.last_layer = weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
    
    def forward(self, z):
        z = self.net(z)
        z = F.normalize(z, dim=-1)
        z = self.last_layer(z)
        return z

class TeacherCenter(nn.Module):
    def __init__(self, out_dim, momentum=0.9):
        super().__init__()

        self.momentum = momentum

        self.register_buffer(
            "center",
            torch.zeros(1, out_dim)
        )

    @torch.no_grad()
    def update(self, teacher_output):

        batch_center = teacher_output.mean(
            dim=(0, 1),
        ).unsqueeze(0)

        self.center.mul_(self.momentum).add_(
            batch_center,
            alpha=1 - self.momentum
        )