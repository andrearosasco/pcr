from torchmetrics import AverageMeter

avg = AverageMeter()

avg(1)
avg(2)
print(avg.compute())
avg(5)
avg(6)

avg(9)
avg(10)

print(avg.compute())