import torch
from torch import nn
from .Transformer import PCTransformer
from .build import MODELS


@MODELS.register_module()
class Hypernetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred

        self.transformer = PCTransformer(in_chans=config.n_channels,
                                         embed_dim=config.embed_dim,
                                         depth=config.encoder_depth,
                                         mlp_ratio=config.mlp_ratio,
                                         qkv_bias=config.qkv_bias,
                                         knn_layer=config.knn_layer,
                                         num_heads=config.num_heads,
                                         attn_drop_rate=config.attn_drop_rate,
                                         drop_rate=config.drop_rate,
                                         qk_scale=config.qk_scale)

        self.output = [[
                nn.Linear(1024, 64 * 3, bias=False).to(config.device),
                nn.Linear(1024, 64, bias=False).to(config.device),
                nn.Linear(1024, 64, bias=False).to(config.device)]]

        for _ in range(2):
            self.output.append([
                    nn.Linear(1024, 64 * 64, bias=False).to(config.device),
                    nn.Linear(1024, 64, bias=False).to(config.device),
                    nn.Linear(1024, 64, bias=False).to(config.device)
                ])

        self.output.append([
            nn.Linear(1024, 64, bias=False).to(config.device),
            nn.Linear(1024, 1, bias=False).to(config.device),
            nn.Linear(1024, 1, bias=False).to(config.device),
        ])

    def forward(self, xyz):
        global_feature = self.transformer(xyz)  # B M C and B M 3
        impl = []
        for layer in self.output:
            impl.append([ly(global_feature) for ly in layer])

        return impl


class ImplicitFunction(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, points):
        x = points

        weights, scales, biases = self.params[0]
        weights = weights.reshape(3, 64)
        scales = scales.squeeze()
        biases = biases.squeeze()

        x = torch.mm(x, weights) * scales + biases
        x = self.dropout(x)
        x = self.relu(x)

        for layer in self.params[1:-1]:
            weights, scales, biases = layer

            weights = weights.reshape(64, 64)
            scales = scales.squeeze()
            biases = biases.squeeze()

            x = torch.mm(x, weights) * scales + biases
            x = self.dropout(x)
            x = self.relu(x)

        weights, scales, biases = self.params[-1]

        weights = weights.reshape(64, 1)
        scales = scales.squeeze()
        biases = biases.squeeze()

        x = torch.mm(x, weights) * scales + biases

        return x
