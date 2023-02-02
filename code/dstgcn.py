from typing import List
import torch
from torch_geometric.nn import GCNConv
from torch import nn
import torch.nn.functional as F

class ConvGraphNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes: List[int], output_dim=1):
        super(ConvGraphNet, self).__init__()
        # Input layer
        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_size))
            else:
                self.layers.append(GCNConv(hidden_sizes[i - 1], hidden_size))
        self.layers.append(GCNConv(hidden_sizes[-1], output_dim))

    def forward(self, x, edge_index):
        x = F.dropout(x)
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))

        return x

class StackedSBlocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(StackedSBlocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        h, g = input
        for module in self[:-1]:
            h = h + module(h, g)
        h = self[-1](h, g)
        return h
        
class fully_connected_layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize fullyConnectedNet.

        Parameters
        ----------
        input_size – The number of expected features in the input x  -> scalar
        hidden_size – The numbers of features in the hidden layer h  -> list
        output_size  – The number of expected features in the output x  -> scalar

        input -> (batch, in_features)

        :return
        output -> (batch, out_features)

        """
        super(fully_connected_layer, self).__init__()

        self.input_size = input_size
        # list
        self.hidden_size = hidden_size
        self.output_size = output_size
        fcList = []
        reluList = []
        for index in range(len(self.hidden_size)):
            if index != 0:
                input_size = self.hidden_size[index - 1]
            fc = nn.Linear(input_size, self.hidden_size[index])
            setattr(self, f'fc{index}', fc)
            fcList.append(fc)
            relu = nn.ReLU()
            setattr(self, f'relu{index}', relu)
            reluList.append(relu)
        self.last_fc = nn.Linear(self.hidden_size[-1], self.output_size)

        self.fcList = nn.ModuleList(fcList)
        self.reluList = nn.ModuleList(reluList)

    def forward(self, input_tensor):

        """
        :param input_tensor:
            2-D Tensor  (batch, input_size)

        :return:
            2-D Tensor (batch, output_size)
            output_tensor
        """
        for idx in range(len(self.fcList)):
            out = self.fcList[idx](input_tensor)
            out = self.reluList[idx](out)
            input_tensor = out
        # (batch, output_size)
        output_tensor = self.last_fc(input_tensor)

        return output_tensor

class STBlock(nn.Module):
    def __init__(self, f_in: int, f_out: int):
        """
        :param f_in: the number of dynamic features each node before
        :param f_out: the number of dynamic features each node after
        """
        super(STBlock, self).__init__()
        # stack four middle layers to transform features from f_in to f_out
        self.spatial_embedding = ConvGraphNet(f_in, [(f_in * (4 - i) + f_out * i) // 4 for i in (1, 4)], f_out)
        self.temporal_embedding = nn.Conv1d(f_out, f_out, 3, padding=1)

    def forward(self, temporal_features: torch.Tensor, edges):
        """
        :param g: batched graphs,
             with the total number of nodes is `node_num`,
             including `batch_size` disconnected subgraphs
        :param temporal_features: shape [node_num, f_in, t_in]
        :return: hidden features after temporal and spatial embedding, shape [node_num, f_out, t_in]
        """
        sp_embedding = self.spatial_embedding(temporal_features, edges)
        return self.temporal_embedding(sp_embedding.transpose(-2, -1))


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(StackedSTBlocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        h, g = input
        for module in self:
            temp = module(h, g)
            h = torch.cat((h, temp.transpose(-2, -1)), dim=2)

        return h

class DSTGCN(nn.Module):
    def __init__(self, f_1=103, f_2=2, f_3=22):
        """
        :param f_1: the number of spatial features each node, default 22
        :param f_2: the number of dynamic features each node, default 1
        :param f_3: the number of features overall (external)
        """
        super(DSTGCN, self).__init__()

        self.spatial_embedding = fully_connected_layer(f_1, [40], 30)

        self.spatial_gcn = StackedSBlocks([ConvGraphNet(30, [30, 30, 30], 30),
                                           ConvGraphNet(30, [30, 30, 30], 30),
                                           ConvGraphNet(30, [28, 26, 14, 22], 20)])
        self.temporal_embedding = StackedSTBlocks([STBlock(f_2, 2), STBlock(4, 4), STBlock(8, 8), STBlock(16, 16)])

        self.temporal_agg = nn.AvgPool1d(12)

        self.external_embedding = fully_connected_layer(f_3, [(f_3 * (4 - i) + 10 * i) // 4 for i in (1, 4)], 20)

        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(20 + 2 + 20, 2), #  + 20
                                          ) # nn.Sigmoid()

    def forward(self,
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                external_features: torch.Tensor,
                edges):
        """
        get predictions
        :param bg: batched graphs,
             with the total number of nodes is `node_num`,
             including `batch_size` disconnected subgraphs
        :param spatial_features: shape [node_num, F_1]
        :param temporal_features: shape [node_num, F_2, T]
        :param external_features: shape [batch_size, F_3]
        :return: a tensor, shape [batch_size], with the prediction results for each graphs
        """
        # print(spatial_features.shape)
        # print(temporal_features.shape)
        # print(external_features.shape)
        s_out = self.spatial_gcn(self.spatial_embedding(spatial_features), edges)

        
        temporal_embeddings = self.temporal_embedding(temporal_features, edges)

        # # t_out of shape [1, node_num, 10]
        t_out = self.temporal_agg(temporal_embeddings)
        t_out.squeeze_()

        e_out = self.external_embedding(external_features)

        output_features = torch.cat((s_out, t_out, e_out), -1) # t_out,

        return self.output_layer(output_features)