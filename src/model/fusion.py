import torch
import torch.nn as nn
from constants import HIDDEN_DIM
    
class GeneTokenEmbedding(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.register_buffer("gene_ids", torch.arange(n_genes))

        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        rna_emb,
        snv_emb,
        cnv_emb,
    ):
        # (G, H) - broadcast to (B, G, H)
        gene_emb = self.gene_embedding(self.gene_ids)

        gene_tokens = gene_emb + rna_emb + snv_emb + cnv_emb

        return self.layernorm(gene_tokens)