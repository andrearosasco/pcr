import torch
from torch import nn
from .Transformer import PCTransformer
from .build import MODELS


@MODELS.register_module()
class Hypernetwork(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query

        self.base_model = PCTransformer(in_chans=3, embed_dim=self.trans_dim, depth=[6, 8], drop_rate=0.,
                                        num_query=self.num_query, knn_layer=self.knn_layer)

        self.output = []

        # self.output.append([
        #         nn.Linear(1024, 64 * 3, bias=False).to(config.device),
        #         nn.Linear(1024, 64, bias=False).to(config.device),
        #         nn.Linear(1024, 64, bias=False).to(config.device),
        #     ])

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
        global_feature = self.base_model(xyz)  # B M C and B M 3
        impl = []
        for layer in self.output:
            impl.append([l(global_feature) for l in layer])

        return impl


class ImplicitFunction:

    def __init__(self, params):
        self.params = params
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

    def __call__(self, points):
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
