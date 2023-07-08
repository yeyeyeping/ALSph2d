import torch
from torch import nn

ce = nn.CrossEntropyLoss()

a = torch.randn(16, 4, 96, 96)
l = torch.empty(16, 96, 96, dtype=torch.long).random_(0, 3)
ce(a, l)
