import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .Transformer import PCTransformer


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size + output_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class HyperNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)

    def forward(self, main_in, sec_in, object_id=None):
        fast_weights, _ = self.backbone(main_in, object_id=object_id)
        return self.sdf(sec_in, fast_weights)


class BackBone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.knn_layer = config.knn_layer

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

        # Select between deep feature extractor and not
        if config.use_deep_weights_generator:
            generator = MLP
        else:
            generator = nn.Linear

        # Select the right dimension for linear layers
        if config.use_object_id:
            global_size = config.out_size + config.n_classes
        else:
            global_size = config.out_size

        # Generate first weight, bias and scale of the input layer of the implicit function
        self.output = nn.ModuleList([nn.ModuleList([
                generator(global_size, config.hidden_dim * 3),
                generator(global_size, config.hidden_dim),
                generator(global_size, config.hidden_dim)])])

        # Generate weights, biases and scales of the hidden layers of the implicit function
        for _ in range(2):
            self.output.append(nn.ModuleList([
                    generator(global_size, config.hidden_dim * config.hidden_dim),
                    generator(global_size, config.hidden_dim),
                    generator(global_size, config.hidden_dim)
                ]))
        # Generate weights, biases and scales of the output layer of the implicit function
        self.output.append(nn.ModuleList([
            generator(global_size, config.hidden_dim),
            generator(global_size, 1),
            generator(global_size, 1),
        ]))

        # self.transformer.apply(self._init_weights)
        for parameter in self.transformer.parameters():
            if len(parameter.size()) > 2:
                torch.nn.init.xavier_uniform_(parameter)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight.data, gain=1)
    #
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, xyz, object_id=None):
        # xyz = torch.reshape(xyz, (xyz.shape[0], -1))
        # global_feature = self.test(xyz)

        global_feature = self.transformer(xyz)  # B M C and B M 3

        if object_id is not None:
            global_feature = torch.cat((global_feature, object_id), dim=-1)

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
        if params is not None:
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
