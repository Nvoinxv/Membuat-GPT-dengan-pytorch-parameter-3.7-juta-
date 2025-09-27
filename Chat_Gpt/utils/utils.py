import torch
import torch.nn as nn

def init_weights_xavier(model):
    """
    Inisialisasi bobot model pakai Xavier uniform
    """
    for name, param in model.named_parameters():
        if param.dim() > 1:  # weight matrix
            nn.init.xavier_uniform_(param)
        else:  # bias
            nn.init.zeros_(param)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def save_checkpoint(model, optimizer, epoch=0, path="checkpoint.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path="checkpoint.pt", device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, resuming at epoch {epoch}")
    return epoch
