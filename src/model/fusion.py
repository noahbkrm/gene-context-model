# Main block with merging token, multi-head attention pooling (1-head), query pooling
import torch
import torch.nn as nn
from constants import HIDDEN_DIM
    
class GeneTokenEmbedding(nn.Module):
    def __init__(self, n_genes: int, rna_encoder, snv_encoder, cnv_encoder, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.register_buffer("gene_ids", torch.arange(n_genes))

        self.rna_encoder = rna_encoder
        self.snv_encoder = snv_encoder
        self.cnv_encoder = cnv_encoder

        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        rna_x, rna_mask,
        snv_x, snv_mask,
        cnv_x, cnv_mask,
    ):
        # (G, H) - broadcast to (B, G, H)
        gene_emb = self.gene_embedding(self.gene_ids)

        rna_emb = self.rna_encoder(rna_x, rna_mask)
        snv_emb = self.snv_encoder(snv_x, snv_mask)
        cnv_emb = self.cnv_encoder(cnv_x, cnv_mask)

        gene_tokens = gene_emb + rna_emb + snv_emb + cnv_emb

        return self.layernorm(gene_tokens)