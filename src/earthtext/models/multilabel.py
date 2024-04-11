import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def activation_from_str(activation_str):
    if activation_str == 'relu':
        return nn.ReLU()

    if activation_str == 'elu':
        return nn.ELU()

    if activation_str == 'sigmoid':
        return nn.Sigmoid()
    
    raise ValueError(f"unknown activation function string '{activation_str}'")

class MultilabelModel(nn.Module):
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

        layers.append(nn.Linear(layers_spec[-1], output_dim))
        layers.append(activation_from_str('sigmoid'))
        
        self.layers = nn.Sequential(*layers)
        
    
    def forward(self, x):
        x = self.layers(x)        
        return x
        
