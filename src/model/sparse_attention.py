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

def prepare_neighbor_index(neighbor_index, x, N):
    B = x.size(0)
    k = neighbor_index.size(-1)

    neighbor_index = neighbor_index.unsqueeze(0).expand(B, -1, -1)
    batch_index = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, k)

    return x[batch_index, neighbor_index]

class SparseAttention(nn.Module):
    def __init__(self, dropout, n_genes, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.n_genes = n_genes

    def reset_neighbors(self):
        self.neighbor_index = create_attention_matrix(
            self.n_genes,
            K_NEIGHBOR
        ).to(self.neighbor_index.device)
    
    def forward(self, input_tokens): # input token embedding dims: (batch, n_tokens, hidden_dim)
        N = input_tokens.size(1)

        graph = prepare_neighbor_index(
            self.neighbor_index,
            input_tokens,
            N
        )

        Q = self.Wq(input_tokens) # Q: q*Wq    dims: (batch, n_genes, hidden_dim)
        Keys = self.Wk(graph) # K: x*Wk    dims: (batch, n_genes, k_neighbors, hidden_dim)
        V = self.Wv(graph) # V: x*Wv    dims: (batch, n_genes, k_neighbors, hidden_dim)
        
        sim_matrix = (
            Q.unsqueeze(2) * Keys
        ).sum(-1)

        d = Q.size(-1)
        sim_matrix = sim_matrix / math.sqrt(d)

        alpha = torch.softmax(sim_matrix, -1) 
        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training
        )

        output = (
            alpha.unsqueeze(-1) * V
        ).sum(dim=2)

        return self.layernorm(output)
        