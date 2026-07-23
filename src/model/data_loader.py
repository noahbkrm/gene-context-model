import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../data/processed")


def load_processed_data():
    return {
        "clinical": pd.read_parquet(DATA_DIR / "clinical.parquet"),
        "cnv": pd.read_parquet(DATA_DIR / "cnv.parquet"),
        "snv": pd.read_parquet(DATA_DIR / "snv.parquet"),
        "rna": pd.read_parquet(DATA_DIR / "rna.parquet"),
    }

def validate_data(data):
    for name, df in data.items():
        print(name)
        print("shape:", df.shape)
        print("unique patients:", df.index.nunique())
        print("duplicate patients:", df.index[df.index.duplicated()].tolist()[:10])
        print()

    clinical = data["clinical"]
    cnv = data["cnv"]

    print("Clinical first 10:")
    print(clinical.index[:10])

    print("\nCNV first 10:")
    print(cnv.index[:10])


def remove_duplicate_patients(data):
    cleaned = {}

    for name, df in data.items():
        before = len(df)

        df = df[~df.index.duplicated(keep="first")]

        after = len(df)

        cleaned[name] = df

    return cleaned

def align_patients(data):
    patients = (
        data["clinical"].index
        .intersection(data["cnv"].index)
        .intersection(data["snv"].index)
        .intersection(data["rna"].index)
    )

    for name in data:
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

    data = load_processed_data()
    data = remove_duplicate_patients(data)
    data = align_patients(data)
    data = align_genes(data)

    if cohort == "full":
        return data

    elif cohort == "debug":
        return reduce_genes(data, n_genes=500)

    else:
        raise ValueError(
            f"Unknown cohort: {cohort}"
        )

def reduce_genes(data, n_genes):

    rna = data["rna"]

    variance = rna.var(axis=0)

    top_genes = (
        variance
        .sort_values(ascending=False)
        .head(n_genes)
        .index
    )

    for name in ["rna", "cnv", "snv"]:
        data[name] = data[name][top_genes]

    data["gene_names"] = top_genes.tolist()

    return data

if __name__ == "__main__":
    dataset = return_dataset("debug")