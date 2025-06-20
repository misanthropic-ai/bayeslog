import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Create a simple tensor
x = torch.randn(3, 3)
print(f"Created tensor: {x.shape}")

# Try MPS if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x_mps = x.to(device)
    print(f"Moved tensor to MPS: {x_mps.device}")