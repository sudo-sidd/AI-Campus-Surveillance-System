import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

# Get the name of the GPU
if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU name:", gpu_name)
else:
    print("No GPU detected.")
