from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl


class EdgeHomophily(nn.Module):
    """Calculate the edge-level homophily representation between node v and its neighbor u.

        Input:
            h_prev_v: the hidden representation of node v in previous layer
            h_prev_u: the hidden representation of neighbor u in previous layer

        Output:
            MLP(concat([h_prev_v, h_prev_u))
    """

    def __init__(self, in_features: int, out_features: int):
        """ A MLP layer to represent edge-level homphily

        Parameters
        ----------
        in_features : int
            Dimension of previous feature vector
        out_features : int
            Dimension of output feature vector
        """
        super(EdgeHomophily, self).__init__()
        self.MLP = nn.Linear(in_features * 2, out_features)

    def forward(self, h_prev_v: torch.Tensor, h_prev_u: torch.Tensor):
        """ Forward computation

        Parameters
        ----------
        h_prev_v : Tensor
            The previous feature representation of node v
        h_prev_u : Tensor
            The previous feature representation of node u
        """

        edge_homophily = self.MLP(torch.cat([h_prev_v, h_prev_u], dim=1))

        return edge_homophily


class NodeAggregator(nn.Module):
    """Aggregate the edge-level homophily representation to get the new node representation.

        Input:
            v: the node v
            neighbors: the neighbors of node v

        Output:
            Aggregated node representation: sum of all edge-level homophily representations
    """

    def __init__(self, in_features: int, out_features: int):
        """ A node aggregator to aggregate edge-level homophily representations

        Parameters
        ----------
        in_features : int
            Dimension of previous feature vector
        out_features : int
            Dimension of output feature vector
        """
        super(NodeAggregator, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, v: torch.Tensor, neighbors: List[torch.Tensor]):
        """ Forward computation

        Parameters
        ----------
        v : Tensor
            The feature representation of node v
        neighbors : List[Tensor]
            A list of feature representations of neighbors of node v
        """
        edges_homophily = EdgeHomophily(self.in_features, self.out_features)

        edge_homophily_representations = []
        for neighbor in neighbors:
            edge_homophily_representations.append(edges_homophily(v, neighbor))

        edge_homophily_representations = torch.stack(
            edge_homophily_representations)

        return torch.sum(edge_homophily_representations, dim=0)
