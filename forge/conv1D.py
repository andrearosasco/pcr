import torch
from torch import nn

# Conv1D with kernel 1 is analogous to Linear
# But technically a convolutional network with a conv1d at the end instead of
# a linear layer, can support image of variable size.
conv = nn.Conv1d(3, 8, 1)

w = conv.weight.detach().clone().permute(2, 1, 0)
b = conv.bias.detach().clone()

input = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.],
                      [10., 11., 12],
                      [13., 14., 15],
                      [16., 17., 18]])

input = input.permute(1, 0)
input = input.unsqueeze(0)
# Input structure: N (batch size) Cin (input channels) (L sequence length)
# In DGCNN the dimensions are 1, 3, 2048

output1 = conv(input)
output2 = torch.bmm(input.permute(0, 2, 1), w) + b

print(output1.round)
print(output2.permute(0, 2, 1))