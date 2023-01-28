from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import DeepGCNLayer, GENConv

# adapted from https://gist.github.com/sparticlesteve/62854712aed7a7e46b70efaec0c64e4f
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, output_dim=2):
        super(RecurrentGCN, self).__init__()
        self.layers = torch.nn.ModuleList([
            GConvGRU(node_features, 256, 1),
            GConvGRU(256, 128, 1),
            GConvGRU(128, 64, 1),
            GConvGRU(64, 32, 1),
            GConvGRU(32, output_dim, 1),
        ])

    def forward(self, graphs, edge_index):
        hidden_states = [None] * len(self.layers)
        predictions = []
        for node_features in graphs:
            hidden_states[0] =  self.layers[0](node_features, edge_index, H=hidden_states[0])
            for i in range(1, len(self.layers)):
                hidden_states[i] = F.relu(self.layers[i](hidden_states[i-1], edge_index, H=hidden_states[i]))
            predictions += [hidden_states[-1]]
        predictions = torch.stack(predictions)
        return predictions

class ConvGraphNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ConvGraphNet, self).__init__()
        # Input layer
        self.layers = nn.ModuleList([
            GCNConv(input_dim, 1024),
            GCNConv(1024, 512),
            GCNConv(512, 256),
            GCNConv(256, 64),
            GCNConv(64, output_dim),
        ])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x

class DeeperGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.node_encoder = nn.Linear(num_features, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)
        
        self.final = GENConv(hidden_channels, out_channels, aggr='softmax')

    def forward(self, x, edge_index):
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.final(x, edge_index)

        return x