import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import activation_from_str


class MultilabelModel(nn.Module):
    """
    assumes an input of shape [batch_size, c(=768)]
    """
    def __init__(self, input_dim, output_dim, layers_spec = [10], activation_fn='relu'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_spec = layers_spec
        self.activation_fn = activation_fn

        layers = [
                nn.Linear(input_dim, layers_spec[0]),
                activation_from_str(activation_fn)
        ]

        for i in range(len(layers_spec)-1):
            layers.append(nn.Linear(layers_spec[i], layers_spec[i+1]))
            layers.append(activation_from_str(activation_fn))

        layers.append(nn.Linear(layers_spec[-1], output_dim))
        layers.append(activation_from_str('sigmoid'))

        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)        
        return x



class DoubleConv2d(nn.Module):
    """ Two-layer Conv2d with a relu activation in between. """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        '''
        Args:
            in_channels: int, number of input channels.
            out_channels: int, number of output channels.
            kernel_size: int, height/width of the kernel.
        '''
        super().__init__()

        padding = kernel_size // 2
        self.doubleconv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
        )

     def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor (B, in_channels, H, W).
        Returns:
            Tensor (B, out_channels, H, W).
        '''
        return self.doubleconv2d(x)



class ContextualCNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, layers_spec = [10], activation_fn='relu') -> None:

        super().__init__()

        # aliases
        i, o = input_dim, output_dim

        self.encoder = DoubleConv2d(in_channels=i, out_channels=i, kernel_size=3)  # encodes the entire neighborhood
        self.final  = MultilabelModel(input_dim=i, output_dim=o, layers_spec=layers_spec, activation_fn=activation_fn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor (B, C, H, W), batch of images.
        Returns:
            Tensor (B, out_channels)
        """
        x = self.encoder(x)     # (B, out_channels, H, W)
        x = x.mean(dim=(2, 3))  # (B, out_channels) global average pooling
        x = self.final(x)       # (B, output_dim)
        return x



class ContextualAvg(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, layers_spec = [10], activation_fn='relu') -> None:

        super().__init__()

        # aliases
        i, o = input_dim, output_dim

        self.final  = MultilabelModel(input_dim=2*i, output_dim=o, layers_spec=layers_spec, activation_fn=activation_fn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor (B, C, H, W), batch of images.
        Returns:
            Tensor (B, out_channels)
        """
        x2 = x.mean(dim=(2, 3))  # (B, out_channels) global average pooling
        x = torch.concat([x, x2], dim=0)  # concatenate avg neighbors embedding w chip embedding
        x = self.final(x)        # (B, output_dim)
        return x