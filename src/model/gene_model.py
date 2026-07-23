import torch.nn as nn
from constants import HIDDEN_DIM

from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding
from rna_encoder import RnaEmbedding
from fusion import GeneTokenEmbedding
from transformer import TransformerEncoder
from projection import Projection
from mask import GeneMask

class GeneModel(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.cnv_encoder =  CNVEmbedding(hidden_dim)
        self.snv_encoder = SNVEmbedding(hidden_dim)
        self.rna_encoder = RnaEmbedding(hidden_dim)
        self.combine_tokens = GeneTokenEmbedding(n_genes, hidden_dim)
        self.projection = Projection(hidden_dim)
        self.mask = GeneMask(hidden_dim)

        self.transformer = TransformerEncoder()

    def forward(self, batch): # image_binary: True is student, False is teacher

        rna_tokens = self.rna_encoder(
            batch["rna_expression"],
            batch["rna_mask"],
        )

        snv_tokens = self.snv_encoder(
            batch["snv_states"],
            batch["snv_mask"]
        )

        cnv_tokens = self.cnv_encoder(
            batch["cnv_states"],
            batch["cnv_mask"]
        )

        token_emb = self.combine_tokens(rna_tokens, snv_tokens, cnv_tokens)

        masked_emb, gene_mask = self.mask(token_emb)

        transformed_emb = self.transformer(masked_emb)

        proj_emb = self.projection(transformed_emb)

        return {
            "projection": proj_emb,
            "embedding": transformed_emb,
            "mask": gene_mask
        }