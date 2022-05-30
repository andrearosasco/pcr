import torch
from configs import server_config
from model import PCRNetwork


# # Load model
from model.Refiner import Refiner

model = PCRNetwork.load_from_checkpoint('./checkpoint/final', config=server_config.ModelConfig)
model.cuda()
model.backbone.cuda()
model.eval()
model.backbone.eval()

fast_weights, _ = model.backbone(torch.randn((1, 2024, 3)).cuda())
# fast_weights = [weight for layer in fast_weights for weight in layer]

x = torch.tensor(torch.randn(1, 10000, 3).cpu().detach().numpy() * 1, requires_grad=True, device='cuda')

refiner = Refiner()
torch.onnx.export(refiner, (x, fast_weights), 'refiner.onnx', input_names=['points'] + [f'param{i}' for i in range(12)],
                  output_names=[f'output'], opset_version=11) # [f'param{i}' for i in range(12)] + ['features']
# x = torch.randn((1, 3, 128)).cuda()
# torch.onnx.export(Simple(), x, 'production/test/simple.onnx', input_names=['input'], output_names=['idx'])