import torch
import torch.nn as nn


import torch

torch.manual_seed(42)

# ----------------------------
# Tiny toy problem
# ----------------------------

BATCH_SIZE = 2
N_GENES = 6
HIDDEN_DIM = 4
N_NEIGHBORS = 3

# Create easily identifiable embeddings
x = torch.arange(
    BATCH_SIZE * N_GENES * HIDDEN_DIM,
    dtype=torch.float32
).reshape(BATCH_SIZE, N_GENES, HIDDEN_DIM)

print("Input tensor shape:", x.shape)
print()

print("Patient 0:")
print(x[0])

print()

print("Patient 1:")
print(x[1])

# ----------------------------
# Fixed neighborhood graph
# ----------------------------

neighbor_index = torch.tensor([
    [0,1,2],   # Gene 0 attends to 0,1,2
    [1,0,4],   # Gene 1 attends to 1,0,4
    [2,3,5],   # Gene 2 attends to 2,3,5
    [3,4,2],   # Gene 3 attends to 3,4,2
    [4,5,1],   # Gene 4 attends to 4,5,1
    [5,2,0],   # Gene 5 attends to 5,2,0
])

print("\nNeighbor index:")
print(neighbor_index)

print("\nShapes")
print("Embeddings:", x.shape)
print("Neighbor index:", neighbor_index.shape)

print(neighbor_index.shape) #(n_genes, k)

# my input tensor x has shape (batch, n_genes, hidden_dim)

neighbor_index = neighbor_index.unsqueeze(0)
print(neighbor_index.shape) # shape: (1, n_genes, k)

neighbor_index = neighbor_index.expand(BATCH_SIZE, -1, -1)
print(neighbor_index.shape) # shape: (BATCH_SIZE, n_genes, k)

batch_index = (
    torch.arange(BATCH_SIZE)
    .view(BATCH_SIZE, 1, 1)
    .expand(-1, N_GENES, N_NEIGHBORS)
)

neighbors = x[
    batch_index,
    neighbor_index
]
print(neighbors.shape)
print(neighbors)

scores = (
    query.unsqueeze(2) * keys
).sum(-1)