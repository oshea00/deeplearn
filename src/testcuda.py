import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"\n✅ CUDA is working!")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Create a tensor and move it to GPU
    device = torch.device("cuda")
    x = torch.randn(3, 3)
    x_cuda = x.to(device)
    
    # Perform a simple operation
    y_cuda = x_cuda @ x_cuda.T
    
    print(f"\nDevice: {x_cuda.device}")
    print(f"Result:\n{y_cuda.cpu()}")
else:
    print("\n❌ CUDA is NOT available")
    print("Possible reasons:")
    print("- No NVIDIA GPU present")
    print("- GPU drivers not installed")
    print("- Running on macOS (CUDA not supported)")
