import pandas as pd
import mygene

def get_clinical_data():
    clinical = pd.read_csv(
        "../../data/Survival_SupplementalTable_S1_20171025_xena_sp",
        sep="\t",
        index_col=0
    )
    return clinical

def get_cnv_data():
    cnv = pd.read_csv(
        "../../data/TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
        sep="\t",
        compression="gzip",
        index_col=0
    )
    cnv = cnv.T
    return cnv

def get_snv_data():
    snv = pd.read_csv(
        "../../data/mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz",
        sep="\t",
        compression="gzip",
        index_col=0
    )
    snv = snv.T
    return snv

def get_rna_data():
    rna = pd.read_csv(
        "../../data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
        sep="\t",
        compression="gzip",
        index_col=0
    )

    # 1. Query MyGene using a string list of your original Entrez IDs
    mg = mygene.MyGeneInfo()
    entrez_ids = rna.index.astype(str).tolist()
    results = mg.querymany(entrez_ids, scopes="entrezgene", fields="symbol", species="human")

    # 2. CRITICAL: Construct a dictionary map matching the original 'query' to the new 'symbol'
    # If a gene symbol is missing, we preserve the original Entrez ID as a fallback
    id_map = {item["query"]: item.get("symbol", item["query"]) for item in results if "query" in item}

    # 3. Safely map the index directly using the dictionary (prevents list-shifting bugs)
    rna.index = rna.index.astype(str).map(id_map)
    rna = rna.groupby(rna.index).mean()
    # 4. Transpose the matrix if your downstream pipeline expects Rows = Samples, Columns = Genes
    rna_transposed = rna.T
    return rna_transposed

def tcga_patient_id(sample):
    return "-".join(sample.split("-")[:3])

if __name__ == "__main__":
    clinical = get_clinical_data()
    clinical.index = clinical.index.map(tcga_patient_id)
    cnv = get_cnv_data()
    cnv.index = cnv.index.map(tcga_patient_id)
    snv = get_snv_data()
    snv.index = snv.index.map(tcga_patient_id)
    rna = get_rna_data()
    rna.index = rna.index.map(tcga_patient_id)

    # Filter for common patients
    patients = (
        rna.index
        .intersection(cnv.index)
        .intersection(snv.index)
        .intersection(clinical.index)
    )

    print(len(patients))

    clinical = clinical.loc[patients]
    rna = rna.loc[patients]
    cnv = cnv.loc[patients]
    snv = snv.loc[patients]

    # Filter for common genes
    common_genes = sorted(
        set(rna.columns)
        & set(cnv.columns)
        & set(snv.columns)
    )

    rna = rna[common_genes]
    cnv = cnv[common_genes]
    snv = snv[common_genes]

    print(len(common_genes))

    clinical.to_parquet("../../data/processed/clinical.parquet")
    rna.to_parquet("../../data/processed/rna.parquet")
    cnv.to_parquet("../../data/processed/cnv.parquet")
    snv.to_parquet("../../data/processed/snv.parquet")