import torch
from torch.utils.data import TensorDataset, DataLoader

class TCGA_Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        
        self.patient_ids = data["clinical"].index.tolist()
        self.gene_names = data["gene_names"]


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


    def __len__(self):
        return len(self.rna)


    def __getitem__(self, idx):

        return {
            "patient_id": self.patient_ids[idx],
            "rna": self.rna[idx],
            "snv": self.snv[idx],
            "cnv": self.cnv[idx]
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