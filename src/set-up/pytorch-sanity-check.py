"""Scanity check ti veryfy that pytorch is installed correctly."""
import torch
import torchvision

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print("CUDA Available: ", torch.cuda.is_available())
