import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from constants import HIDDEN_DIM

class SNVEmbedding(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.snv_state_embedding = nn.Embedding(2, hidden_dim) # 0, 1, NA
        self.missing_bias = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.missing_bias, std = 0.02)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.register_buffer(  # Static so it won't be treated as a parameter to be optimized
            "gene_ids",
            torch.arange(n_genes)
        )
    
    @staticmethod
    def prepare(df: pd.DataFrame) -> torch.Tensor:

        df = df.copy()

        observed_mask = (~df.isna()).astype(int)

        df = df.fillna(0).astype("int64") # fill missing values as 3rd item (index 2)
        snv_state_tensor = torch.from_numpy(df.to_numpy(dtype=np.int64))
        mask_tensor = torch.tensor(observed_mask.values, dtype=torch.int64)
        return snv_state_tensor, mask_tensor
    
    def forward(self, snv_state_tensor, mask):

        gene_emb = self.gene_embedding(self.gene_ids)
        state_emb = self.snv_state_embedding(snv_state_tensor)

        was_missing = (1 - mask).unsqueeze(-1)
        state_emb = state_emb + was_missing * self.missing_bias

        combined_emb = gene_emb + state_emb
        return self.layernorm(combined_emb)

if __name__ == "__main__":

    print()