import torch
from configs import server_config
from model import PCRNetwork


# # Load model
model = PCRNetwork.load_from_checkpoint('./checkpoint/final', config=server_config.ModelConfig)
model.cuda()
model.eval()

x = torch.ones((1, 2024, 3)).cuda()
torch.onnx.export(model.backbone, x, 'pcr.onnx', input_names=['input'], output_names=['features'] + [f'param{i}' for i in range(12)], opset_version=11)