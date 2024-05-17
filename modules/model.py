import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from modules.mlp_layer import CustomLinear, CustomMLP
from modules.regularization import CustomBatchNorm1d


class GNN_layer(nn.Module):
    """ A GNN layer for the ConsisG4D model. 
        (Corresponding to the MySimpleConv_MR_test class in the ConsisGAD model)

    The GNN layer consists of:
        - an input layer of linear transformation, shape `(in_feats, out_feats)`
        - an output layer of linear transformation, shape `(out_feats, out_feats)`
        - an edge projection MLPs for each edge type
        - an output projection MLPs for the message passing
        - a batch normalization layer for the edge types

        The GNN layer is used to propagate messages on the graph and update the node features.
    """

    def __init__(self, in_feats: int, out_feats: int, e_types: list, drop_rate: float = 0.0,
                 mlp3_dim: int = 64, bn_type: int = 0):
        """ Initialize the GNN layer.
        Args:
            `in_feats`: the input feature dimension
            `out_feats`: the output feature dimension
            `e_types`: the list of edge types
            `drop_rate`: the dropout rate
            `mlp3_dim`: the hidden dimension of the edge projection MLPs, default: 64
            `bn_type`: the batch normalization type, default: 0
        """
        super(GNN_layer, self).__init__()
        self.e_types = e_types
        self.drop_rate = drop_rate
        self.bn_type = bn_type
        self.mlp3_dim = mlp3_dim
        """
            `multi_relation`: whether the graph contains multiple types of relations
            The amazon dataset contains 3 edge types:
                - U-P-U (users reviewing at least one same product),
                - U-S-U (users having at least one same star rating within one week)
                - U-V-U (users with top-5% mutual review TF-IDF similarities)
        """
        self.multi_relation = len(self.e_types) > 1

        self.proj_edges = nn.ModuleDict()  # a dictionary mapping each edge type to an MLP
        # Initialize the MLPs for each edge type
        for e_t in self.e_types:
            self.proj_edges[e_t] = CustomMLP(in_feats * 2, out_feats,
                                             p=self.drop_rate, hid_dim=self.mlp3_dim)

        # Initialize the output projection MLP (for the predictor)
        self.proj_out = CustomLinear(
            out_feats, out_feats, bias=True)  # output projection MLP
        # Initialize the skip connection MLP
        if in_feats != out_feats:
            # to match the input and output dimensions
            self.proj_skip = CustomLinear(in_feats, out_feats, bias=True)
        else:
            # do nothing if the dimensions are the same
            self.proj_skip = nn.Identity()
        # the sum of the projected output and the skip connection is the final output

        if self.bn_type in [2, 3]:  # Amazon dataset uses batch normalization type 2
            # a dictionary mapping each edge type to a batch normalization layer
            self.edge_bn = nn.ModuleDict()
            # Initialize the batch normalization layers for each edge type
            for e_t in self.e_types:
                self.edge_bn[e_t] = CustomBatchNorm1d(out_feats)

    def udf_edges(self, e_t: str):
        """ User-defined message passing function for the edges based on the edge type.

        Args:
            `e_t`: the edge type

        Returns:
            `func`: the message passing function for the edges

        """
        assert e_t in self.e_types, 'Invalid edge type'

        # get the relevant MLP for this edge type
        edge_func = self.proj_edges[e_t]

        def func(edges):
            # Concatenate the source node v and destination neightbor node v features
            msg = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
            # Apply the MLP to the concatenated features
            msg = edge_func(msg)
            return {'msg': msg}

        return func

    def forward(self, g: dgl.DGLGraph, features, update_bn: bool = True):
        """ Forward pass of the GNN layer. 
        Args:
            `g`: the DGL graph
            `features`: the input node features
            `update_bn`: whether to update the batch normalization layer
        Returns:
            `out`: the output node features
        """

        # local_scope: make sure that the changes are not propagated to the original graph
        # local as in local variables (only within this function, not global variables)
        with g.local_scope():
            # initialize the source and destination node features equal to the input features
            # `src_feats`: source node features
            # `dst_feats`: destination node features
            src_feats = dst_feats = features

            # if the graph is a block, update the destination node features
            # a block is a subgraph of the original graph
            if g.is_block:
                dst_feats = src_feats[:g.num_dst_nodes()]

            # set the source and destination node features in the graph
            # `h` as in hidden features
            g.srcdata['h'] = src_feats
            g.dstdata['h'] = dst_feats

            # apply the user-defined message passing function for each edge type
            for e_t in g.etypes:
                g.apply_edges(self.udf_edges(e_t), etype=e_t)

            # apply batch normalization if the batch normalization type is 2 or 3
            if self.bn_type in [2, 3]:
                if not self.multi_relation:
                    # if there is only one edge type, apply batch normalization for the 1st edge type
                    # which is the only edge type
                    # update_running_stats: whether to update the running statistics (mean and variance) during training
                    g.edata['msg'] = self.edge_bn[self.e_types[0]](g.edata['msg'], update_running_stats=update_bn)
                else:
                    # if there are multiple edge types, apply batch normalization for each edge type
                    for e_t in g.canonical_etypes:  # iterate over all unique edge types
                        g.edata['msg'][e_t] = self.edge_bn[e_t[1]](g.edata['msg'][e_t], update_running_stats=update_bn)

            # define the message passing function for each edge type
            etype_dict = {}
            for e_t in g.etypes:
                etype_dict[e_t] = (
                    # copy the edge features to the compute message field
                    # which means the message passed from node v to node u is the `msg` field 
                    # from the above `udf_edges` function
                    fn.copy_e('msg', out='msg'),
                    # sums up the msg field from all neighbor, saved to the out field
                    fn.sum('msg', 'out'))

            # update all nodes with the message passing function for each edge type
            # cross_reducer: the cross-reducer function to aggregate messages from different edge types
            # stack: stack the messages from different edge types
            g.multi_update_all(etype_dict=etype_dict, cross_reducer='stack')

            # get the aggregated messages from the destination nodes
            out = g.dstdata.pop('out')
            # sum up the aggregated messages along the feature dimension
            # which means that the messages from different edge types are summed up
            out = torch.sum(out, dim=1)

            # apply the output projection MLP and the skip connection MLP
            out = self.proj_out(out) + self.proj_skip(dst_feats)

            # return the output node features
            return out


class GNN_backbone(nn.Module):
    """ The GNN backbone model for the ConsisG4D model.
        (Corresponding to the simpleGNN_MR class in the ConsisGAD model)

    The GNN model consists of:
        - an input projection MLP layer
        - a list of GNN layers
        - a list of batch normalization layers
        - an output projection MLP layer

        The GNN model is used to learn the node embeddings on the graph.
    """

    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int,
                 num_layers: int, e_types: list, 
                 input_drop: float, hidden_drop: float, mlp_drop: float,
                 mlp12_dim: int, mlp3_dim: int, bn_type: int):
        """ Initialize the GNN model.
        Args:
            `in_feats`: the input feature dimension
            `hidden_feats`: the hidden feature dimension
            `out_feats`: the output feature dimension
            `num_layers`: the number of GNN layers
            `e_types`: the list of edge types
            `input_drop`: the dropout rate for the input features
            `hidden_drop`: the dropout rate for the hidden features (right after the input layer)
            `mlp_drop`: the dropout rate for the MLPs
            `mlp12_dim`: the hidden dimension of the input projection MLP and output projection MLP
            `mlp3_dim`: the hidden dimension of the GNN layer MLPs
            `bn_types`: the batch normalization types
        """
        super(GNN_backbone, self).__init__()
        self.gnn_list = nn.ModuleList()  # a list of GNN layers
        self.bn_list = nn.ModuleList()  # a list of batch normalization layers
        self.num_layers = num_layers
        self.input_drop = input_drop
        self.hidden_drop = hidden_drop
        self.mlp_drop = mlp_drop
        self.mlp12_dim = mlp12_dim
        self.mlp3_dim = mlp3_dim
        self.bn_types = bn_type
        
        # the input projection MLP
        self.proj_in = CustomMLP(
            in_feats, hidden_feats, p=self.hidden_drop, hid_dim=self.mlp12_dim)
        in_feats = hidden_feats  # update the input feature dimension

        self.in_bn = None  # initialize the batch normalization layer for the input features
        if self.bn_types in [1, 3]:  # Amazon dataset uses batch normalization type 2
            self.in_bn = CustomBatchNorm1d(hidden_feats)
        
        # add GNN layers to the list
        for i in range(num_layers):
            # if the first layer, use the input feature dimension
            # otherwise, use the hidden feature dimension 
            in_dim = in_feats if i == 0 else hidden_feats

            # add a GNN layer to the list
            self.gnn_list.append(
                GNN_layer(in_feats=in_dim, out_feats=hidden_feats,
                          e_types=e_types, drop_rate=self.mlp_drop,
                          mlp3_dim=self.mlp3_dim, bn_type=self.bn_types))
            
            # add a batch normalization layer to the list
            self.bn_list.append(CustomBatchNorm1d(hidden_feats))

        # the output projection MLP
        self.proj_out = CustomMLP(
            hidden_feats*(num_layers+1), out_feats, p=self.mlp_drop,
            hid_dim=self.mlp12_dim, final_act=False)
        
        # dropout layer
        self.dropout = nn.Dropout(p=self.hidden_drop)
        # dropout layer for the input features
        self.dropout_in = nn.Dropout(p=self.input_drop) 
        # activation function
        self.activation = F.selu

    def forward(self, blocks: list, update_bn: bool = True, return_logits: bool = False):
        """ Forward pass of the GNN model.
        Args:
            `blocks`: the list of DGL blocks
            A block is a graph consisting of two sets of nodes: the source nodes and destination nodes.
            The source nodes are the nodes in the previous layer, and the destination nodes are the nodes in the current layer.
            `update_bn`: whether to update the batch normalization layer
            `return_logits`: whether to return the logits
        Returns:
            `inter_results`: the intermediate results
        """
        
        # the number of destination nodes in the final block
        final_num = blocks[-1].num_dst_nodes()  
        # the input node features
        h = blocks[0].srcdata['feature']
        # dropout for the input features
        h = self.dropout_in(h)

        # pass the input features through the input projection MLP
        h = self.proj_in(h)

        # apply batch normalization for the input features
        if self.in_bn is not None:
            h = self.in_bn(h, update_running_stats=update_bn)
        # a list to store the intermediate results
        inter_results = []
        # append the input features to the intermediate results sliced up to the final number of nodes
        # e.g., if the final number of nodes is 100, append the first 100 nodes of the input features
        inter_results.append(h[:final_num]) 

        # iterate over the blocks and the GNN layers
        for block, gnn, bn in zip(blocks, self.gnn_list, self.bn_list):
            # pass the input features through the GNN layer
            h = gnn(block, h, update_bn)
            # apply batch normalization
            h = bn(h, update_running_stats=update_bn)
            # apply the activation function
            h = self.activation(h)
            # apply dropout
            h = self.dropout(h)
            # append the intermediate results sliced up to the final number of nodes
            inter_results.append(h[:final_num])

        # if return_logits is True, return the intermediate results
        if return_logits:
            return inter_results
        else:
            # stack the intermediate results along the feature dimension
            h = torch.stack(inter_results, dim=1)
            # flatten the intermediate results to the final number of nodes
            h = h.reshape(h.shape[0], -1)
            # pass the intermediate results through the output projection MLP
            h = self.proj_out(h)
            # return the output node features, log softmax to produce log probabilities for each class
            return h.log_softmax(dim=-1)