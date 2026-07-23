import torch.nn as nn
from constants import HIDDEN_DIM

# Process
# 1. Perform layernorm on input tokens (x)
# 2. Feed this into a multihead attention block (recall: Q= qWq, K = kWk, V = vWv, sim = Q*K.T, softmax(sim), sim*V/sqrt(d))
# 3. Add x + attention
# 4. Perform a second layernorm
# 5. Feed into our feed-forward network
# 6. Return x + ffn output

class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        n_heads: int = 8, # Revisit later
        mlp_ratio: int = 4,
        dropout: float = 0.1, # Revisit later
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm1 = self.norm1(x)

        attn_out, attn_weights = self.attn(
            x_norm1,
            x_norm1,
            x_norm1,
            need_weights=False,
        )

        x = x + attn_out

        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)

        x = x + ffn_out

        return x

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock(),
        ])
    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x