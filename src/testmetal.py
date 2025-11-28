import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    # Create a tensor and move it to MPS device
    device = torch.device("mps")
    x = torch.randn(3, 3)
    x_mps = x.to(device)
    
    # Perform a simple operation
    y_mps = x_mps @ x_mps.T
    
    print(f"\n✅ Metal/MPS is working!")
    print(f"Device: {x_mps.device}")
    print(f"Result:\n{y_mps.cpu()}")
else:
    print("\n❌ Metal/MPS is NOT available")
    print("You may be using a CPU-only build of PyTorch")
