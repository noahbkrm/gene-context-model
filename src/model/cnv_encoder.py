
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from constants import HIDDEN_DIM

class CNVEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.cnv_state_embedding = nn.Embedding(5, hidden_dim) # -2, -1, 0, 1, 2
        self.missing_bias = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.missing_bias, std = 0.02)
        self.layernorm = nn.LayerNorm(hidden_dim)
    
    @staticmethod
    def prepare(df: pd.DataFrame) -> torch.Tensor:

        df = df.copy()

        observed_mask = (~df.isna()).astype(int)

        df = df + 2 # Shift all values to be >= 0
        df = df.fillna(0).astype("int64") # Placeholder value; missing handled by mask
        cnv_state_tensor = torch.from_numpy(df.to_numpy(dtype=np.int64).copy())
        mask_tensor = torch.from_numpy(observed_mask.to_numpy(dtype=np.float32).copy())
        return cnv_state_tensor, mask_tensor
    
    def forward(self, cnv_state_tensor, mask):

        state_emb = self.cnv_state_embedding(cnv_state_tensor)

        was_missing = (1 - mask).unsqueeze(-1)
        state_emb = state_emb + was_missing * self.missing_bias

        return self.layernorm(state_emb)

if __name__ == "__main__":

    print()