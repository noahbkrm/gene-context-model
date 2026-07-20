import torch
from torch.utils.data import Dataset, DataLoader
from google.cloud import bigquery
import pandas as pd
import numpy as np

class PanCancerOmicDataset(Dataset):
    def __init__(self, limit_samples=1000):
        # Initialize Google BigQuery Client
        self.client = bigquery.Client()
        
        print("Streaming molecular matrices from cloud into memory...")
        self.samples, self.rna, self.cnv, self.snv = self._load_data(limit_samples)
        
        # Convert NumPy matrices directly into PyTorch Tensors
        self.rna_tensor = torch.tensor(self.rna, dtype=torch.float32)
        self.cnv_tensor = torch.tensor(self.cnv, dtype=torch.float32)
        self.snv_tensor = torch.tensor(self.snv, dtype=torch.float32)

    def _load_data(self, limit):
        # 1. Fetch Unfiltered RNA Expression (RSEM Values)
        rna_query = f"""
        SELECT sample_barcode, gene_name, rsem_gene_expression
        FROM `isb-cgc-bq.TCGA_versioned.rna_seq_v2_expression_gdc_2024_03`
        LIMIT {limit * 100}
        """
        rna_df = self.client.query(rna_query).to_dataframe()
        rna_pivot = rna_df.pivot_table(index='sample_barcode', columns='gene_name', values='rsem_gene_expression').fillna(0)
        
        # 2. Fetch Unfiltered Copy Number Variations (CNV)
        cnv_query = """
        SELECT sample_barcode, gene_name, copy_number
        FROM `isb-cgc-bq.TCGA_versioned.cnv_gdc_2024_03`
        WHERE sample_barcode IN UNNEST(@barcodes)
        """
        barcodes = rna_pivot.index.tolist()
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("barcodes", "STRING", barcodes)]
        )
        cnv_df = self.client.query(cnv_query, job_config=job_config).to_dataframe()
        cnv_pivot = cnv_df.pivot_table(index='sample_barcode', columns='gene_name', values='copy_number').fillna(0)
        
        # 3. Fetch Somatic Mutations (SNV Binary Matrix)
        snv_query = """
        SELECT sample_barcode, Hugo_Symbol, Variant_Classification
        FROM `isb-cgc-bq.TCGA_versioned.somatic_mutation_gdc_2024_03`
        WHERE sample_barcode IN UNNEST(@barcodes)
        """
        snv_df = self.client.query(snv_query, job_config=job_config).to_dataframe()
        snv_df['mutated'] = 1
        snv_pivot = snv_df.pivot_table(index='sample_barcode', columns='Hugo_Symbol', values='mutated').fillna(0)
        
        # Align all DataFrames to ensure perfectly matching patient rows
        common_samples = rna_pivot.index.intersection(cnv_pivot.index).intersection(snv_pivot.index)
        
        return (
            common_samples.tolist(),
            rna_pivot.loc[common_samples].values,
            cnv_pivot.loc[common_samples].values,
            snv_pivot.loc[common_samples].values
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Maps features perfectly by patient index
        return {
            "sample_id": self.samples[idx],
            "rna": self.rna_tensor[idx],
            "cnv": self.cnv_tensor[idx],
            "snv": self.snv_tensor[idx]
        }

# Instantiate the dataset completely in-memory
dataset = PanCancerOmicDataset(limit_samples=500)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Validation Check
first_batch = next(iter(dataloader))
print("Aligned Batch Sample Size:", first_batch["rna"].shape[0])
print("RNA Feature Tensor Shape:", first_batch["rna"].shape)
print("CNV Feature Tensor Shape:", first_batch["cnv"].shape)
print("SNV Feature Tensor Shape:", first_batch["snv"].shape)
