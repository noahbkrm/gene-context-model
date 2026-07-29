import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *
import math
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *

def create_attention_matrix(n_genes: int, k: int):
    neighbor_index = torch.empty(
        n_genes,
        k,
        dtype=torch.long
    )

    for gene in range(n_genes):

        allowed = torch.cat([
            torch.arange(gene),
            torch.arange(gene + 1, n_genes)
        ])

        random_neighbors = allowed[
            torch.randperm(len(allowed))[:k - 1]
        ]

        neighbor_index[gene] = torch.cat([
            torch.tensor([gene]),
            random_neighbors
        ])

    return neighbor_index


def gather_neighbors(x, neighbor_index):
    B, N, H = x.shape
    K = neighbor_index.size(1)

    neighbor_index = neighbor_index.unsqueeze(0).expand(B, -1, -1)

    batch_index = (
        torch.arange(B, device=x.device)
        .view(B, 1, 1)
        .expand(-1, N, K)
    )

    return x[batch_index, neighbor_index]


class SparseAttention(nn.Module):
    def __init__(self, dropout, n_genes, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

        self.layernorm = nn.LayerNorm(hidden_dim)

        self.dropout = dropout
        self.n_genes = n_genes

        self.register_buffer(
            "neighbor_index",
            create_attention_matrix(n_genes, K_NEIGHBOR)
        )

    def reset_neighbors(self):

        self.neighbor_index.copy_(
            create_attention_matrix(
                self.n_genes,
                K_NEIGHBOR
            ).to(self.neighbor_index.device)
        )

    def forward(self, input_tokens):

        B, N, H = input_tokens.shape

        Q = self.Wq(input_tokens)          # (B, N, H)
        K_all = self.Wk(input_tokens)      # (B, N, H)
        V_all = self.Wv(input_tokens)      # (B, N, H)

        K = gather_neighbors(
            K_all,
            self.neighbor_index
        )                                 # (B, N, K, H)

        V = gather_neighbors(
            V_all,
            self.neighbor_index
        )                                 # (B, N, K, H)

        scores = (
            Q.unsqueeze(2) * K
        ).sum(dim=-1)

        scores = scores / math.sqrt(H)

        alpha = torch.softmax(
            scores,
            dim=-1
        )

        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training
        )

        output = (
            alpha.unsqueeze(-1) * V
        ).sum(dim=2)

        return self.layernorm(output)