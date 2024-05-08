import torch
import torch.nn as nn

class CustomBatchNorm1d(nn.BatchNorm1d):
    """ A custom batch normalization layer with optional running statistics update."""
    def forward(self, input: torch.Tensor, update_running_stats: bool = False) -> torch.Tensor:
        """ Forward pass of the batch normalization layer.
        Args:
            `input`: the input tensor
            `update_running_stats`: whether to update the running statistics
        Returns:
            `output`: the output tensor
        """
        self.track_running_stats = update_running_stats
        return super(CustomBatchNorm1d, self).forward(input)
    