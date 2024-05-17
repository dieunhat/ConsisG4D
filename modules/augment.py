import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import dgl
    
class SoftAttentionDrop(nn.Module): # learnable masking module
    def __init__(self, config):
        super(SoftAttentionDrop, self).__init__()        
        """
        Args:
            hidden_dim: dimension of input tensor
            temperature: sharpening temperature
            drop_ratio: drop ratio of sharpen function        
        """
        self.temp = config['temperature'] # lower temperature -> sharper mask
        self.drop_ratio = config['drop-ratio'] # drop ratio of sharpen function
        self.hidden_dim = config['hidden-dim']
        self.mask_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.mask_projection.weight.data.fill_(0)
        # self.epsilon = epsilon
    
    def sharpen(self, input, drop_ratio, input_dims, temperature):
        """ 
        Args: 
            input: input tensor
            drop_ratio: drop ratio
            input_dims: dimension of input tensor
            temperature: sharpening temperature
        Returns:
            sharpened tensor
        """
        epsilon = 1e-12
        input_ = torch.zeros(input.shape)

        for i in range(round(drop_ratio * input_dims)):
            m = (1. - input_)
            mask = torch.log(m + epsilon)
            y = torch.softmax((mask - input)/temperature, dim=1)
            input_ += y * m
        
        return (1. - input_)
    
    def forward(self, feature, in_eval=False):
        mask = self.mask_projection(feature) # attention function   
        sharpened_mask = self.sharpen(mask, self.drop_ratio, mask.shape[1], self.temp)

        if in_eval:
            sharpened_mask = sharpened_mask.detach()
        
        return feature * sharpened_mask
    
def fixed_augmentation(graph, seed_nodes, sampler, aug_type, epsilon=0.0):
    # assert augmentation in ['dropout', 'dropnode', 'dropedge', 'replace', 'drophidden', 'none']
    """
    Args:
        graph: input graph
        seed_nodes: random nodes in the graph to be augmented on
        sampler: sampler (dgl.dataloading.MultiLayerFullNeighborSampler()
        aug_type: augmentation type
        epsilon: epsilon value (small epsilon value as noise injection)
    """
    with graph.local_scope(): # apply augmentation to the subgraph only
        if aug_type == 'dropedge':
            del_edges = {}
            for et in graph.etypes:
                _, _, eid = graph.in_edges(seed_nodes, etype=et, form='all')
                num_remove = math.floor(eid.shape[0] * epsilon)
                del_edges[et] = eid[torch.randperm(eid.shape[0])][:num_remove]
            aug_graph = graph
            for et in del_edges.keys():
                aug_graph = dgl.remove_edges(aug_graph, del_edges[et], etype=et)

            input_nodes, output_nodes, blocks = sampler.sample_blocks(aug_graph, seed_nodes)

        else:
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)

            if aug_type == 'dropout':
                blocks[0].srcdata['feature'] = F.dropout(blocks[0].srcdata['feature'], epsilon)
                
            elif aug_type == 'dropnode':
                blocks[0].srcdata['feature'] = blocks[0].remove_nodes(blocks[0].srcdata['feature'], epsilon)
                
            else:
                pass
            
    return input_nodes, output_nodes, blocks