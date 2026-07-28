import torch

def gpu_mem(label):
    print(
        f"{label:20s}"
        f" allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB"
        f" reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB"
    )