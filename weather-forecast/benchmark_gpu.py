import torch
import time

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create large random tensors on CPU and GPU
size = 10000
x_cpu = torch.randn(size, size)
x_gpu = torch.randn(size, size, device=device)

# Measure CPU time
start = time.time()
y_cpu = x_cpu @ x_cpu  # Matrix multiplication on CPU
cpu_time = time.time() - start

# Measure GPU time
start = time.time()
y_gpu = x_gpu @ x_gpu  # Matrix multiplication on GPU
torch.cuda.synchronize()  # Ensure GPU computation finishes
gpu_time = time.time() - start

print(f"CPU Time: {cpu_time:.4f} sec")
print(f"GPU Time: {gpu_time:.4f} sec (should be much faster)")