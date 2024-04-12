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
