import torch
from torch import nn
from .Transformer import PCTransformer


class HyperNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)

    def forward(self, main_in, sec_in):
        fast_weights, _ = self.backbone(main_in)
        return self.sdf(sec_in, fast_weights)



class BackBone(nn.Module):
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
                                         qk_scale=config.qk_scale,
                                         out_size=config.out_size)

        self.output = [[
                nn.Linear(config.out_size, config.hidden_dim * 3, bias=True).to(config.device),
                nn.Linear(config.out_size, config.hidden_dim, bias=True).to(config.device),
                nn.Linear(config.out_size, config.hidden_dim, bias=True).to(config.device)]]

        for _ in range(2):
            self.output.append([
                    nn.Linear(config.out_size, config.hidden_dim * config.hidden_dim, bias=True).to(config.device),
                    nn.Linear(config.out_size, config.hidden_dim, bias=True).to(config.device),
                    nn.Linear(config.out_size, config.hidden_dim, bias=True).to(config.device)
                ])

        self.output.append([
            nn.Linear(config.out_size, config.hidden_dim, bias=True).to(config.device),
            nn.Linear(config.out_size, 1, bias=True).to(config.device),
            nn.Linear(config.out_size, 1, bias=True).to(config.device),
        ])

        # self.test = nn.Linear(2048*3, 1024)

    def forward(self, xyz):
        # xyz = torch.reshape(xyz, (xyz.shape[0], -1))
        # global_feature = self.test(xyz)

        global_feature = self.transformer(xyz)  # B M C and B M 3
        fast_weights = []
        for layer in self.output:
            fast_weights.append([ly(global_feature) for ly in layer])

        return fast_weights, global_feature


class ImplicitFunction(nn.Module):

    def __init__(self, config, params=None):
        super().__init__()
        self.params = params
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)
        self.hidden_dim = config.hidden_dim

    def set_params(self, params):
        self.params = params

    def forward(self, points, params=None):
        if self.params is None:
            self.params = params

        if self.params is None:
            raise ValueError('Can not run forward on uninitialized implicit function')

        x = points
        # TODO: I just added unsqueeze(1), reshape(-1) and bmm and everything works (or did I introduce some kind of bug?)
        weights, scales, biases = self.params[0]
        weights = weights.reshape(-1, 3, self.hidden_dim)
        scales = scales.unsqueeze(1)
        biases = biases.unsqueeze(1)

        x = torch.bmm(x, weights) * scales + biases
        x = self.dropout(x)
        x = self.relu(x)

        for layer in self.params[1:-1]:
            weights, scales, biases = layer

            weights = weights.reshape(-1, self.hidden_dim, self.hidden_dim)
            scales = scales.unsqueeze(1)
            biases = biases.unsqueeze(1)

            x = torch.bmm(x, weights) * scales + biases
            x = self.dropout(x)
            x = self.relu(x)

        weights, scales, biases = self.params[-1]

        weights = weights.reshape(-1, self.hidden_dim, 1)
        scales = scales.unsqueeze(1)
        biases = biases.unsqueeze(1)

        x = torch.bmm(x, weights) * scales + biases

        return x
