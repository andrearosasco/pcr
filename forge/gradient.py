import torch
from torch import nn

conv = nn.Conv1d(3, 8, 1)

input = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.],
                      [10., 11., 12],
                      [13., 14., 15],
                      [16., 17., 18]], requires_grad=True)

input = input.permute(1, 0)
input = input.unsqueeze(0)

print(input.requires_grad)
output = conv(input)
print(output.requires_grad)


