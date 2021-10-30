import torch
from torchmetrics import Accuracy

acc = Accuracy()

pred = torch.tensor([0.6, 0.2])
gt = torch.tensor([1, 1])

print(acc(pred, gt))

pred = torch.tensor([0.6, 0.2])
gt = torch.tensor([1, 0])


print(acc(pred, gt))
print(acc.compute())