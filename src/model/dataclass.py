import torch
from torch.utils.data import DataLoader
from rna_encoder import RnaStats, RnaEmbedding
from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding

class TCGA_Dataset(torch.utils.data.Dataset):

    def __init__(self, data, rna_stats: RnaStats):
        self.rna = torch.tensor(
            data["rna"].values,
            dtype=torch.float32
        )
            
        self.snv = torch.tensor(
            data["snv"].values,
            dtype=torch.float32
        )

        self.cnv = torch.tensor(
            data["cnv"].values,
            dtype=torch.float32
        )

        self.patient_ids = data["clinical"].index.tolist()
        self.gene_names = data["gene_names"]

        rna_expression, rna_mask = RnaEmbedding.prepare(self.rna, rna_stats)
        cnv_states, cnv_mask = CNVEmbedding.prepare(self.cnv)
        snv_states, snv_mask = SNVEmbedding.prepare(self.snv)

        self.rna_expression = rna_expression
        self.rna_mask = rna_mask
        self.cnv_states = cnv_states
        self.cnv_mask = cnv_mask
        self.snv_states = snv_states
        self.snv_mask = snv_mask

    def __len__(self):
        return len(self.rna)


    def __getitem__(self, idx):

        return {
            "patient_id": self.patient_ids[idx],
            "gene_names": self.gene_names[idx],
            "rna_expression": self.rna_expression[idx],
            "rna_mask": self.rna_mask[idx],
            "snv_states": self.snv_states[idx],
            "snv_mask": self.snv_mask[idx],
            "cnv_states": self.cnv_states[idx],
            "cnv_mask": self.cnv_mask[idx],
        }

def get_loader(dataset, shuffle: bool = True):
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        pin_memory=True
    )
    return loader

if __name__ == "__main__":

    from data_loader import return_dataset

    data = return_dataset("debug")

    dataset = TCGA_Dataset(data)

    loader = get_loader(dataset)

    batch = next(iter(loader))

    print(batch["rna"].shape)
    print(batch["snv"].shape)
    print(batch["cnv"].shape)
    print(batch["patient_id"][:5])

    idx = 0

    print(batch["patient_id"][idx])
    print(batch["rna"][idx][:10])
    print(batch["snv"][idx][:10])
    print(batch["cnv"][idx][:10])