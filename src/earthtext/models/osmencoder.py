import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import activation_from_str

class OSMEncoder(nn.Module):
    """
    assumes an input of shape [batch_size, h, w, 2, 2]
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

        # just linear output
        layers.append(nn.Linear(layers_spec[-1], output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    
    def forward(self, x):
        x = self.layers(x)        
        return x
        
