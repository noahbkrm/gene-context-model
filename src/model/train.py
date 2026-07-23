import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from constants import *
from gene_model import GeneModel, GeneTokenizer
from rna_encoder import RnaStats, RnaEmbedding
from dataclass import TCGA_Dataset, get_loader
from data_loader import return_dataset
from mask import GeneMask
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_models(
        n_genes: int,
        hidden_dim: int = HIDDEN_DIM,
    ):

    student_tokenizer = GeneTokenizer(
        n_genes,
        hidden_dim,
    ).to(device)

    teacher_tokenizer = copy.deepcopy(student_tokenizer)

    masker = GeneMask(hidden_dim).to(device)
    
    student_model = GeneModel(hidden_dim).to(device)
    teacher_model = copy.deepcopy(student_model).to(device) # Set teacher to match student
    
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(student_tokenizer.parameters()) +
        list(student_model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-2,
    )

    return teacher_model, student_model, teacher_tokenizer, student_tokenizer, optimizer, masker

def update_teacher_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    ema_param: float = EMA_PARAM,
    ):
    with torch.no_grad():
        for teacher_param, student_param in zip(
            teacher_model.parameters(),
            student_model.parameters()
        ):
            teacher_param.data = (  #thetaT = thetaT*m + (1-m)thetaC
                ema_param * teacher_param.data
                +
                (1 - ema_param) * student_param.data
            )

def training(
        teacher_model: GeneModel,
        student_model: GeneModel,
        teacher_tokenizer: GeneTokenizer,
        student_tokenizer: GeneTokenizer,
        loader: DataLoader,
        optimizer,
        scaler,
        masker,
        ema_param: float = EMA_PARAM,
    ):

    student_model.train()
    student_tokenizer.train()
    teacher_model.eval()
    teacher_tokenizer.eval()
    total_loss = 0

    progress = tqdm(loader, desc="Training", leave=False)

    for batch in progress:

        optimizer.zero_grad()

        batch_gpu = {}

        for k,v in batch.items():

            if torch.is_tensor(v):
                batch_gpu[k] = v.to(device)

            else:
                batch_gpu[k] = v
        
        with torch.amp.autocast("cuda"):

            student_raw_tokens = student_tokenizer(batch_gpu)

            student_tokens, student_mask = masker(student_raw_tokens)

            teacher_tokens = teacher_tokenizer(batch_gpu)

            with torch.no_grad():
                teacher = teacher_model(teacher_tokens) # Calculate teacher

            student = student_model(student_tokens)

            z_teacher = teacher["projection"]
            z_student = student["projection"]

            # Apply softmax and temp, then calculate cross-entropy loss
            teacher_probs = torch.softmax(z_teacher/T_TEACHER, dim=-1)
            student_log_probs = torch.log_softmax(z_student/T_STUDENT, dim=-1)

            ce = -(teacher_probs * student_log_probs).sum(dim=-1)
            loss = ce[student_mask].mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_teacher_model(teacher_model, student_model, ema_param)
        update_teacher_model(teacher_tokenizer, student_tokenizer, ema_param)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)

    print(
        "Embedding std:",
        z_student.std().item()
    )
    cos = F.cosine_similarity(
        z_student.detach(),
        z_teacher.detach(),
        dim=-1
    ).mean()

    print(f"Cosine similarity: {cos.item():.4f}")

    return avg_loss

def prepare_dataset(cohort: pd.DataFrame, rna_stats: RnaStats):

    dataset = TCGA_Dataset(cohort, rna_stats=rna_stats)
    return dataset

def generate_embeddings(model, loader, device):

    model.eval()
    embeddings = []

    with torch.no_grad():

        for batch in loader:

            batch_gpu = {}

            for k,v in batch.items():

                if torch.is_tensor(v):
                    batch_gpu[k] = v.to(device)

                else:
                    batch_gpu[k] = v

                tokens = student_tokenizer(batch_gpu)

                out = model(tokens)

                embeddings.append(
                    out["embedding"].cpu()
                )

    return torch.cat(embeddings, dim=0), model.gene_names

if __name__ == "__main__":
    train_cohort = return_dataset("debug")
    
    rna_stats = RnaEmbedding.fit(train_cohort["rna"])
    
    train_dataset = prepare_dataset(train_cohort, rna_stats)

    loader = get_loader(train_dataset)

    teacher_model, student_model, teacher_tokenizer, student_tokenizer, optimizer, masker = initialize_models(
        n_genes=rna_stats.n_genes,
        hidden_dim = HIDDEN_DIM
    )

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        loss = training(
            teacher_model,
            student_model,
            teacher_tokenizer,
            student_tokenizer,
            loader,
            optimizer,
            scaler,
            masker
        ) 
        print(f"Epoch Loss: {loss:.4f}") 
    
    torch.save(
        {
            "tokenizer": student_tokenizer.state_dict(),
            "student_model": student_model.state_dict(),
            "teacher_model": teacher_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        "jepa_checkpoint.pt",
    )

    student_model.eval() # Switch student model into eval mode

    for p in student_model.parameters(): # Freeze the student model
        p.requires_grad = False
            
    embedding_loader = get_loader(
        train_dataset,
        shuffle=False
    )

    embeddings, gene_ids = generate_embeddings(
        student_model,
        embedding_loader,
        device
    )

    print(embeddings.shape)

    print(
        "Embedding variance:",
        embeddings.var(dim=0).mean()
    )

    print(
        "Gene similarity:",
        F.cosine_similarity(
            embeddings[0],
            embeddings[1],
            dim=0
        )
    )

    print(embeddings.std())
    print(embeddings.mean())

    idx = torch.randint(0, embeddings.size(0), (1000, 2))

    cosines = torch.stack([
        F.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings[j].unsqueeze(0),
            dim=1
        ).squeeze()
        for i, j in idx
    ])

    print("Mean cosine:", cosines.mean())
    print("Std cosine:", cosines.std())
    print("Min cosine:", cosines.min())
    print("Max cosine:", cosines.max())

    torch.save(embeddings,"gene_embeddings.pt")

    embedding_df = pd.DataFrame(
        embeddings.numpy(),
    )

    embedding_df.index.name = "gene_id"
    embedding_df.to_csv("gene_embeddings.csv")