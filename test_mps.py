import torch
import time

device = torch.device('mps')
X = torch.randn(400, 90, 12).to(device)
y = torch.randn(400, 7).to(device)

# Without device sync
start = time.time()
for _ in range(80):
    indices = torch.randperm(X.size(0), device=device)
    for start_idx in range(0, X.size(0), 128):
        idx = indices[start_idx:start_idx+128]
        X_b = X[idx]
torch.mps.synchronize()
print("With device:", time.time() - start)

# With CPU sync
start = time.time()
for _ in range(80):
    indices = torch.randperm(X.size(0))
    for start_idx in range(0, X.size(0), 128):
        idx = indices[start_idx:start_idx+128]
        X_b = X[idx]
torch.mps.synchronize()
print("Without device (CPU sync):", time.time() - start)
