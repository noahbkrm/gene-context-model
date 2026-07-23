import torch

from data_loader import return_dataset
from dataclass import TCGA_Dataset, get_loader
from rna_encoder import RnaEmbedding
from gene_model import GeneModel


def main():

    # -----------------------------
    # 1. Load debug cohort
    # -----------------------------
    data = return_dataset("debug")

    print("Genes:", len(data["gene_names"]))
    print("Patients:", len(data["clinical"]))


    # -----------------------------
    # 2. Fit RNA normalization stats
    # -----------------------------
    rna_stats = RnaEmbedding.fit(
        data["rna"]
    )


    # -----------------------------
    # 3. Create dataset + loader
    # -----------------------------
    dataset = TCGA_Dataset(
        data,
        rna_stats
    )

    loader = get_loader(dataset)

    batch = next(iter(loader))


    print("\nInput batch shapes:")
    print("RNA:", batch["rna_expression"].shape)
    print("RNA mask:", batch["rna_mask"].shape)

    print("SNV:", batch["snv_states"].shape)
    print("SNV mask:", batch["snv_mask"].shape)

    print("CNV:", batch["cnv_states"].shape)
    print("CNV mask:", batch["cnv_mask"].shape)


    # -----------------------------
    # 4. Create model
    # -----------------------------
    n_genes = len(data["gene_names"])

    model = GeneModel(
        n_genes=n_genes
    )

    model.eval()


    # -----------------------------
    # 5. Forward pass
    # -----------------------------
    with torch.no_grad():

        output, gene_mask = model(batch)


    # -----------------------------
    # 6. Validate outputs
    # -----------------------------
    print("\nModel outputs:")

    print(
        "Projection:",
        output.shape
    )

    print(
        "Gene mask:",
        gene_mask.shape
    )


    # -----------------------------
    # 7. Expected dimensions
    # -----------------------------
    B = batch["rna_expression"].shape[0]
    G = n_genes

    print("\nExpected:")
    print(
        f"Projection should be ({B}, {G}, projection_dim)"
    )

    print(
        f"Mask should be ({B}, {G})"
    )


    # -----------------------------
    # 8. Assertions
    # -----------------------------

    assert output.shape[0] == B
    assert output.shape[1] == G

    assert gene_mask.shape == (B, G)


    print("\nSUCCESS: Forward pass completed")


if __name__ == "__main__":
    main()