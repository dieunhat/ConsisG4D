import torch
import dgl
import math
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class CustomLinear(nn.Linear):
    """ A custom linear layer that is initialized with Xavier normal initialization and zero bias."""

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class CustomMLP(nn.Module):
    """
    A multi-layer perceptron (MLP) module for the ConsisG4D model.

    The MLP module consists of:
        - an input layer of linear transformation, shape `(in_dim, hid_dim)`
        - activation function ELU
        - a normalization layer with normalized shape `(hid_dim)`
        - a dropout layer with dropout rate `p`
        - an output layer of linear transformation, shape `(hid_dim, out_dim)`

        If `final_act` is set to `True`, the activation function and dropout is applied at the end.
    """

    def __init__(self, in_dim: int, out_dim: int, 
                 hid_dim: int = 64, p: float = 0.0, final_act: bool = True):
        """ Initialize the MLP module.

        Args:
            `in_dim`: the input dimension
            `out_dim`: the output dimension
            `hid_dim`: the hidden dimension
            `p`: the dropout rate
            `final_act`: whether to apply the activation function at the end
        """
        super(CustomMLP, self).__init__()

        self.input_layer = CustomLinear(in_dim, hid_dim, bias=True)
        self.activation = nn.ELU()
        self.normalization = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(p=p)
        self.output_layer = CustomLinear(hid_dim, out_dim, bias=True)
        self.final_act = final_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the MLP module.

        Args:
            `x`: the input tensor of shape `(batch_size, in_dim)`
            
        Returns:
            `x`: the output tensor of shape `(batch_size, out_dim)`
        """
        
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.normalization(x)
        x = self.output_layer(x)
        if self.final_act:
            x = self.activation(x)
            x = self.dropout(x)
        return x
