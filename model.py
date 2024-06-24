import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, Linear
from utils import normalize_A, generate_cheby_adj

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} (model.py)".format(device))

class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        result = self.gc[0](x, adj[0])  # Previously gc1
        for i in range(1, len(self.gc)):  # Previously gc1
            result += self.gc[i](x, adj[i])  # Previously gc1
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, in_channels: int, num_electrodes: int, k_adj, out_channels: int, num_classes: int = 3):
        """
        :param in_channels: The feature dimension of each electrode.
        :param num_electrodes: The number of electrodes.
        :param k_adj: The number of graph convolutional layers.
        :param out_channels: The feature dimension of  the graph after GCN.
        :param num_classes: The number of classes to predict.
        """
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes * out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).to(device))
        nn.init.uniform_(self.A, 0.01, 0.5)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # data can also be standardized offline
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)

        result = F.softmax(result, dim=1)*5 + .5  # Added
        return result
