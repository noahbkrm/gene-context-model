import torch.nn as nn
from constants import HIDDEN_DIM, BATCH

from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding
from rna_encoder import RnaEmbedding, RnaStats
from fusion import GeneTokenEmbedding
from transformer import TransformerBlock
from projection import Projection
from mask import GeneMask

class GeneModel(nn.Module):
    def __init__(self, rna_stats: RnaStats, n_genes: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.cnv_encoder =  CNVEmbedding(n_genes, hidden_dim)
        self.snv_encoder = SNVEmbedding(n_genes, hidden_dim)
        self.rna_encoder = RnaEmbedding(rna_stats, hidden_dim)
        self.combine_tokens = GeneTokenEmbedding(hidden_dim)
        self.projection = Projection(hidden_dim)
        self.mask = GeneMask(hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock(),
        ])

    def forward(self, batch, image_binary: bool): # image_binary: True is student, False is teacher

        rna_tokens = self.rna_encoder(
            batch["rna_expression"],
            batch["rna_mask"],
        )

        cnv_tokens = self.cnv_encoder(
            batch["cnv_states"],
        )

        snv_tokens = self.snv_encoder(
            batch["snv_states"],
        )

        if mask_snv:
            snv_tokens = self.snv_masker(snv_tokens)

        token_emb = self.combine_tokens(clinical_tokens, rna_tokens, cnv_tokens, snv_tokens)
        patient_emb = self.attention_pooling(token_emb)

        return patient_emb