import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


# Load embeddings
emb = pd.read_csv(
    "gene_embeddings.csv",
    index_col="gene"
)

# Convert to tensor
X = torch.tensor(
    emb.values,
    dtype=torch.float32
)


# --------------------
# PCA analysis
# --------------------

pca = PCA()

X_pca = pca.fit_transform(
    emb.values
)

pc1_scores = pd.Series(
    X_pca[:, 0],
    index=emb.index
)

print("\nMost negative PC1 genes:")
print(
    pc1_scores.sort_values()
    .head(20)
)

print("\nMost positive PC1 genes:")
print(
    pc1_scores.sort_values()
    .tail(20)
)


# --------------------
# Nearest neighbors
# --------------------

def get_neighbors(
        gene,
        n_neighbors=20
    ):

    if gene not in emb.index:
        print(f"{gene} not found")
        return

    # Query vector
    query = torch.tensor(
        emb.loc[gene].values,
        dtype=torch.float32
    )

    # Normalize embeddings
    embeddings = torch.tensor(
        emb.values,
        dtype=torch.float32
    )

    similarities = F.cosine_similarity(
        query.unsqueeze(0),
        embeddings,
        dim=1
    )

    neighbors = (
        pd.DataFrame({
            "gene": emb.index,
            "similarity": similarities.numpy()
        })
        .sort_values(
            "similarity",
            ascending=False
        )
    )

    # Remove itself
    neighbors = neighbors[
        neighbors["gene"] != gene
    ]

    return neighbors.head(n_neighbors)


# --------------------
# Test known genes
# --------------------

query = [
    "KRT5",
    "KRT14",
    "KRT17",
    "KRT6A",
    "KRT19",
    "SFTPC",
    "MUC1",
    "EPCAM"
]


for gene in query:
    print("\n================")
    print("QUERY:", gene)
    print(get_neighbors(gene))

X = torch.tensor(
    emb.values,
    dtype=torch.float32
)

X = F.normalize(X, dim=1)

similarities = X @ X.T

# remove diagonal
mask = ~torch.eye(
    similarities.shape[0],
    dtype=torch.bool
)

vals = similarities[mask]

print("Mean cosine:", vals.mean())
print("Std:", vals.std())
print("Min:", vals.min())
print("Max:", vals.max())

u,s,v = np.linalg.svd(
    emb.values,
    full_matrices=False
)

effective_rank = (
    np.sum(s)**2 /
    np.sum(s**2)
)

print(effective_rank)

print("Singular values:")
print(s[:20])

print("\nExplained variance:")
variance = s**2 / np.sum(s**2)

print(variance[:20])