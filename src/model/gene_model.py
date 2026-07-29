import torch.nn as nn
import torch
from constants import HIDDEN_DIM

from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding
from rna_encoder import RnaEmbedding
from fusion import GeneTokenEmbedding
from transformer import FullTransformerEncoder, SparseTransformerEncoder
from projection import Projection
from utils import gpu_mem

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
    def __init__(self, n_genes, hidden_dim: int = HIDDEN_DIM, method: str = "Sparse"):
        super().__init__()
        self.projection = Projection(hidden_dim)
        if method == "Sparse":
            self.transformer = SparseTransformerEncoder(n_genes)
        elif method == "Full":
            self.transformer = FullTransformerEncoder()

    def reset_neighbors(self):
        if hasattr(self.transformer, "reset_sparse_neighbors"):
            self.transformer.reset_sparse_neighbors()

    def load_sparse_neighbors_from(self, other):
        for my_block, other_block in zip(
            self.transformer.blocks,
            other.transformer.blocks,
        ):
            my_block.sparse_attn.neighbor_index.copy_(
                other_block.sparse_attn.neighbor_index
            )

    def forward(self, gene_tokens): # image_binary: True is student, False is teacher
        transformed_emb = self.transformer(gene_tokens)

        """ print(
            "Backbone std:",
            transformed_emb.std().item()
        ) """

        proj_emb = self.projection(transformed_emb)

        """ print(
            "Projection std:",
            proj_emb.std().item()
        ) """

        return {
            "projection": proj_emb,
            "embedding": transformed_emb,
        }