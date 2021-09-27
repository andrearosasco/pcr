import torch
from torch import nn

m = nn.Conv1d(16, 33, 3)
input = torch.randn(20, 16, 50)
output = m(input)
print(input.shape)
print(output.shape)