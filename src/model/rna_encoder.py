
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from constants import HIDDEN_DIM
from dataclasses import dataclass

@dataclass
class RnaStats:
    train_gene_mean: pd.Series
    train_gene_std: pd.Series

    @property
    def n_genes(self):
        return len(self.train_gene_mean)

class RnaEmbedding(nn.Module):
    def __init__(self,  stats: RnaStats, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.expression = nn.Linear(1, hidden_dim)
        self.missing_bias = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.missing_bias, std = 0.02)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.gene_embedding = nn.Embedding(stats.n_genes, hidden_dim)
    
    @staticmethod
    def fit(train_df: pd.DataFrame) -> RnaStats: # Fitting RnaStats on training data

        df = train_df.copy()

        # Compute mean and standard deviation as np.ndarray
        train_gene_std = df.std(axis = 0, ddof = 0)
        train_gene_mean = df.mean(axis = 0)

        rna_stats_obj = RnaStats(train_gene_mean= train_gene_mean, train_gene_std=train_gene_std)
        return rna_stats_obj

    @staticmethod
    def prepare(df, stats):

        observed_mask = (~df.isna()).astype(float)

        df = df.fillna(stats.train_gene_mean)

        df = (df - stats.train_gene_mean) / stats.train_gene_std

        expression_tensor = torch.tensor(df.values, dtype=torch.float32)

        mask_tensor = torch.from_numpy(observed_mask.to_numpy(dtype=np.float32).copy())

        return expression_tensor, mask_tensor
    
    def forward(self, expression_tensor, mask):

        expression_emb = self.expression(expression_tensor.unsqueeze(-1))
        was_missing_exp = (1 - mask).unsqueeze(-1)
        expression_emb = expression_emb + was_missing_exp * self.missing_bias

        return self.layernorm(expression_emb)

if __name__ == "__main__":
    print()