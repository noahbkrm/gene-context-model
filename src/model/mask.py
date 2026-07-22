import torch
import torch.nn as nn
from constants import HIDDEN_DIM, MASK_RATIO

class GeneMask(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.mask_embedding = nn.Parameter(
            torch.empty(hidden_dim)
        )
        nn.init.normal_(self.mask_embedding, std=0.02)


    def forward(self, gene_tokens, mask_ratio: int = MASK_RATIO):
        B, G, D = gene_tokens.shape
        mask = (torch.rand(B, G, device=gene_tokens.device) < mask_ratio)

        masked_tokens = gene_tokens.clone()
        masked_tokens[mask] = self.mask_embedding

        return masked_tokens, mask