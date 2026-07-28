import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path("../../data/processed")


def load_processed_data():

    clinical = pd.read_parquet(DATA_DIR / "clinical.parquet")

    cnv = pd.read_parquet(DATA_DIR / "cnv.parquet")

    snv = pd.read_parquet(DATA_DIR / "snv.parquet")

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")

    return {
        "clinical": clinical,
        "cnv": cnv,
        "snv": snv,
        "rna": rna,
    }

def validate_data(data):
    for name, df in data.items():
        print(name)
        if isinstance(df, pd.DataFrame):
            print("shape:", df.shape)
            print("unique patients:", df.index.nunique())
        else:
            print(type(df))
        print()

    clinical = data["clinical"]
    cnv = data["cnv"]

    print("Clinical first 10:")
    print(clinical.index[:10])

    print("\nCNV first 10:")
    print(cnv.index[:10])


def remove_duplicate_patients(data):

    for name, df in data.items():

        if not isinstance(df, pd.DataFrame):
            continue

        mask = ~df.index.duplicated(keep="first")
        data[name] = df.loc[mask]

    return data

def align_patients(data):

    patients = (
        data["clinical"].index
        .intersection(data["cnv"].index)
        .intersection(data["snv"].index)
        .intersection(data["rna"].index)
    )

    for name in ["clinical", "cnv", "snv", "rna"]:
        data[name] = data[name].loc[patients]

    return data

def align_genes(data):
    genes = (
        data["rna"].columns
        .intersection(data["cnv"].columns)
        .intersection(data["snv"].columns)
    )

    genes = sorted(genes)

    for name in ["rna", "cnv", "snv"]:
        data[name] = data[name][genes]

    data["gene_names"] = genes

    return data

def return_dataset(cohort: str):

    print("started return")

    data = load_processed_data()

    print("data loaded")

    if cohort == "debug":
        data = reduce_genes(data, n_genes=1500)
        print("reduced genes")

    data = remove_duplicate_patients(data)
    print("dedup")

    data = align_patients(data)
    data = align_genes(data)

    print("aligned")

    data = remove_low_variance_genes(data)

    print("low")

    print("Memory usage:")

    return data

def reduce_genes(data, n_genes, random_state=21):

    genes = (
        pd.Series(data["rna"].columns)
        .sample(
            n=n_genes,
            random_state=random_state,
            replace=False
        )
        .tolist()
    )

    for name in ["rna", "cnv", "snv"]:
        data[name] = data[name][genes]

    data["gene_names"] = genes

    return data

def remove_low_variance_genes(data, min_variance=1e-8):

    data["rna"] = data["rna"].astype(np.float32)

    variance = np.nanvar(
        data["rna"].values,
        axis=0
    )

    keep = data["rna"].columns[variance > min_variance]

    print(
        f"Keeping {len(keep)} of {len(variance)} genes "
        f"(removed {(variance <= min_variance).sum()} low-variance genes)"
    )

    for name in ["rna", "cnv", "snv"]:
        data[name] = data[name][keep]

    data["gene_names"] = keep.tolist()

    return data

if __name__ == "__main__":
    dataset = return_dataset("debug")