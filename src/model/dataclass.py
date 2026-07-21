import torch

class TCGA_Dataset(torch.utils.data.Dataset):

    def __init__(self, data):

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
            "rna": self.rna[idx],
            "snv": self.snv[idx],
            "cnv": self.cnv[idx]
        }