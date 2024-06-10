import torch
from torch import nn, Tensor
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



class MultisizeContextualCNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, layers_spec = [10], activation_fn='relu') -> None:

        super().__init__()

        # aliases
        i, o = input_dim, output_dim

        self.encoder_1 = nn.Conv2d(in_channels=i, out_channels=i, kernel_size=3, padding=0, groups=i)
        self.encoder_2 = nn.Conv2d(in_channels=i, out_channels=i, kernel_size=5, padding=0, groups=i)
        self.encoder_4 = nn.Conv2d(in_channels=i, out_channels=i, kernel_size=9, padding=0, groups=i)
        self.encoder_6 = nn.Conv2d(in_channels=i, out_channels=i, kernel_size=13, padding=0, groups=i)
        self.encoder_8 = nn.Conv2d(in_channels=i, out_channels=i, kernel_size=17, padding=0, groups=i)

        self.final  = MultilabelModel(input_dim=6*i, output_dim=o, layers_spec=layers_spec, activation_fn=activation_fn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor (B, C, H, W), batch of images.
        Returns:
            Tensor (B, out_channels)
        """
        c = x.shape[-1]//2
        slice_it = lambda x, r: x[:, :, (c - r):(c + r + 1), (c - r):(c + r + 1)]

        x_0 = slice_it(x, 0)  # radius 0
        x_1 = slice_it(x, 1)  # radius 1
        x_2 = slice_it(x, 2)
        x_4 = slice_it(x, 4)
        x_6 = slice_it(x, 6)
        x_8 = x              # radius 8

        w_1 = self.encoder_1(x_1)  # all of them (B, o, 1, 1)
        w_2 = self.encoder_2(x_2)
        w_4 = self.encoder_4(x_4)
        w_6 = self.encoder_6(x_6)
        w_8 = self.encoder_8(x_8)

        x = torch.concat([x_0, w_1, w_2, w_4, w_6, w_8], dim=1).squeeze()  # concatenate neighborhood encodings w chip embedding
        x = self.final(x)       # (B, output_dim)

        return x



class ContextualCNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, layers_spec = [10], activation_fn='relu', channel_specific=True) -> None:

        super().__init__()

        # aliases
        i, o = input_dim, output_dim

        if channel_specific:
            self.encoder = nn.Conv2d(in_channels=i, out_channels=i, kernel_size=3, padding=3//2, groups=i)
        else:
            self.encoder = DoubleConv2d(in_channels=i, out_channels=i, kernel_size=3)  # encodes the entire neighborhood

        self.final  = MultilabelModel(input_dim=2*i, output_dim=o, layers_spec=layers_spec, activation_fn=activation_fn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor (B, C, H, W), batch of images.
        Returns:
            Tensor (B, out_channels)
        """
        c = x.shape[-1]//2

        x2 = self.encoder(x)     # (B, out_channels, H, W)
        x2 = x2.mean(dim=(2, 3))  # (B, out_channels) global average pooling
        x = torch.concat([x[:, :, c, c], x2], dim=1)  # concatenate avg neighbors embedding w chip embedding
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
        c = x.shape[-1]//2

        x2 = x.mean(dim=(2, 3))  # (B, out_channels) global average pooling
        x = torch.concat([x[:, :, c, c], x2], dim=1)  # concatenate avg neighbors embedding w chip embedding
        x = self.final(x)        # (B, output_dim)

        return x
