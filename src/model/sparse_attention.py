import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *
import math

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
            torch.randperm(len(allowed))[:k-1]
        ]

        neighbor_index[gene] = torch.cat([
            torch.tensor([gene]),
            random_neighbors
        ])
    return neighbor_index # Returns shape (n_genes, k)

class SparseAttention(nn.Module):
    def __init__(self, dropout, n_genes, hidden_dim: int = HIDDEN_DIM):
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
        new_neighbors = create_attention_matrix(
            self.n_genes,
            K_NEIGHBOR
        )

        self.neighbor_index.copy_(
            new_neighbors.to(self.neighbor_index.device)
        )
    
    def forward(self, input_tokens):
        """
        input_tokens:
            (B, N, H)
        """

        B, N, H = input_tokens.shape

        # Project everything once
        Q = self.Wq(input_tokens)      # (B, N, H)
        K_all = self.Wk(input_tokens)  # (B, N, H)
        V_all = self.Wv(input_tokens)  # (B, N, H)

        output = torch.empty_like(input_tokens)

        scale = math.sqrt(H)

        for gene in range(N):

            nbrs = self.neighbor_index[gene]      # (K,)

            # Gather projected neighbors
            K = K_all[:, nbrs]                    # (B, K, H)
            V = V_all[:, nbrs]                    # (B, K, H)

            # Query for this gene
            q = Q[:, gene]                        # (B, H)

            scores = (
                q.unsqueeze(1) * K
            ).sum(dim=-1)

            scores = scores / scale

            alpha = torch.softmax(
                scores,
                dim=-1
            )

            alpha = F.dropout(
                alpha,
                p=self.dropout,
                training=self.training,
            )

            attended = (
                alpha.unsqueeze(-1) * V
            ).sum(dim=1)

            output[:, gene] = attended

        return self.layernorm(output)