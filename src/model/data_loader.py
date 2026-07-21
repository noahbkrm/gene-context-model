import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../data/processed")


def load_processed_data():
    data = {
        "clinical": pd.read_parquet(DATA_DIR / "clinical.parquet"),
        "cnv": pd.read_parquet(DATA_DIR / "cnv.parquet"),
        "snv": pd.read_parquet(DATA_DIR / "snv.parquet"),
        "rna": pd.read_parquet(DATA_DIR / "rna.parquet"),
    }

    # Get patients present in all modalities
    patients = (
        data["rna"].index
        .intersection(data["cnv"].index)
        .intersection(data["snv"].index)
        .intersection(data["clinical"].index)
    )

    # Restrict every dataframe to shared patients
    for name in data:
        data[name] = data[name].loc[patients]

    return data


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

        print(
            f"{name}: removed {before-after} duplicates"
        )

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

if __name__ == "__main__":
    data = load_processed_data()
    data = remove_duplicate_patients(data)
    data = align_patients(data)

    validate_data(data)

    for name, df in data.items():
        print(name)
        print(df.shape)
        print(df.head())
        print()