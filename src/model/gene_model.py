import torch.nn as nn
from constants import HIDDEN_DIM

from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding
from rna_encoder import RnaEmbedding
from fusion import GeneTokenEmbedding
from transformer import TransformerEncoder
from projection import Projection

class GeneTokenizer(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        self.rna_encoder = RnaEmbedding(hidden_dim)
        self.snv_encoder = SNVEmbedding(hidden_dim)
        self.cnv_encoder = CNVEmbedding(hidden_dim)

        self.combine_tokens = GeneTokenEmbedding(n_genes, hidden_dim,)

    def forward(self, batch):

        rna_tokens = self.rna_encoder(
            batch["rna_expression"],
            batch["rna_mask"],
        )

        snv_tokens = self.snv_encoder(
            batch["snv_states"],
            batch["snv_mask"],
        )

        cnv_tokens = self.cnv_encoder(
            batch["cnv_states"],
            batch["cnv_mask"],
        )

        gene_tokens = self.combine_tokens(rna_tokens, snv_tokens, cnv_tokens,)

        return gene_tokens

class GeneModel(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.projection = Projection(hidden_dim)
        self.transformer = TransformerEncoder()

    def forward(self, gene_tokens): # image_binary: True is student, False is teacher

        transformed_emb = self.transformer(gene_tokens)

        proj_emb = self.projection(transformed_emb)

        return {
            "projection": proj_emb,
            "embedding": transformed_emb,
        }