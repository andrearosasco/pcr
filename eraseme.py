import torch

x = torch.rand(3, 10)
idx = torch.randint_like(x, 0, 2).bool()
pass
