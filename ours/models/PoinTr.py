import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
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

        self.output = [
            [
                nn.Linear(1024, 64 * 3, bias=False).to(config.device),
                nn.Linear(1024, 64, bias=False).to(config.device),
                nn.Linear(1024, 64, bias=False).to(config.device),
            ]
            for x in range(4)]

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
        # self.bn = nn.BatchNorm1d()

    def __call__(self, points):
        x = points
        l = len(self.params)

        for i in range(l):
            weights, scales, biases = self.params[i]
            x = torch.bmm(x, weights) * scales + biases
            x = self.relu(x)

        weights, scales, biases = self.params[i]
        x = torch.bmm(x, weights) * scales + biases

        return x




class TargetNetwork(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['TN']['use_bias']
        # target network layers out channels
        out_ch = config['model']['TN']['layer_out_channels']

        layer_data, split_index = self._get_layer_data(start_index=0, end_index=out_ch[0] * 3,
                                                       shape=(out_ch[0], 3), weights=weights)
        self.layers = {"1": layer_data}

        for x in range(1, len(out_ch)):
            layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                           end_index=split_index + (out_ch[x - 1] * out_ch[x]),
                                                           shape=(out_ch[x], out_ch[x - 1]), weights=weights)
            self.layers[str(x + 1)] = layer_data

        layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                       end_index=split_index + (out_ch[-1] * 3),
                                                       shape=(3, out_ch[-1]), weights=weights)
        self.output = layer_data
        self.activation = torch.nn.ReLU()
        assert split_index == len(weights)

    def forward(self, x):
        for layer_index in self.layers:
            x = torch.mm(x, torch.transpose(self.layers[layer_index]["weight"], 0, 1))
            if self.use_bias:
                assert "bias" in self.layers[layer_index]
                x = x + self.layers[layer_index]["bias"]
            x = self.activation(x)
        return torch.mm(x, torch.transpose(self.output["weight"], 0, 1)) + self.output.get("bias", 0)

    def _get_layer_data(self, start_index, end_index, shape, weights):
        layer_data = {"weight": weights[start_index:end_index].view(shape[0], shape[1])}
        if self.use_bias:
            layer_data["bias"] = weights[end_index:end_index + shape[0]]
            end_index = end_index + shape[0]
        return layer_data, end_index
