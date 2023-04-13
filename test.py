import torch

input = torch.randn(size=(16, 3, 256, 256))
print(torch.max(input))
