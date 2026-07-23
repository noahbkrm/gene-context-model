import torch
from torch.utils.data import DataLoader
from rna_encoder import RnaStats, RnaEmbedding
from cnv_encoder import CNVEmbedding
from snv_encoder import SNVEmbedding

class TCGA_Dataset(torch.utils.data.Dataset):

    def __init__(self, data, rna_stats: RnaStats):

        self.patient_ids = data["clinical"].index.tolist()
        self.gene_names = data["gene_names"]

        rna_expression, rna_mask = RnaEmbedding.prepare(data["rna"], rna_stats)
        cnv_states, cnv_mask = CNVEmbedding.prepare(data["cnv"])
        snv_states, snv_mask = SNVEmbedding.prepare(data["snv"])

        self.rna_expression = rna_expression
        self.rna_mask = rna_mask
        self.cnv_states = cnv_states
        self.cnv_mask = cnv_mask
        self.snv_states = snv_states
        self.snv_mask = snv_mask

    def __len__(self):
        return len(self.rna_expression)

    def __getitem__(self, idx):

        return {
            "patient_id": self.patient_ids[idx],
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

    rna_stats = RnaEmbedding.fit(data["rna"])

    dataset = TCGA_Dataset(data, rna_stats)

    loader = get_loader(dataset)

    batch = next(iter(loader))

    print(batch["rna_expression"].shape)
    print(batch["snv_mask"].shape)
    print(batch["cnv_states"].shape)
    print(batch["patient_id"][:5])