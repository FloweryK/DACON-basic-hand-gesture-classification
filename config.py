import torch

class TrainerConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 16
    num_workers = 0