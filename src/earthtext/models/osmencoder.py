import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import activation_from_str

class OSMEncoder(nn.Module):
    """
    assumes an input of shape [batch_size, h, w, 2, 2]
    """
    def __init__(self, 
                 output_dim, 
                 layers_spec = [10], 
                 activation_fn='relu', 
                 output_activation_fn=None,
                 osm_tags_indexes = None,
                 osm_number_of_tags = 99,
                 use_osm_counts = True,
                 use_osm_areas = True,
                 use_osm_lengths = True,
        ):
        super().__init__()

        self.output_dim = output_dim
        self.layers_spec = layers_spec
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn
        self.osm_number_of_tags = osm_number_of_tags
        self.osm_tags_indexes = np.r_[osm_tags_indexes] if osm_tags_indexes is not None else None
        self.use_osm_counts = use_osm_counts
        self.use_osm_areas = use_osm_areas
        self.use_osm_lengths = use_osm_lengths

        if self.use_osm_counts + self.use_osm_areas + self.use_osm_lengths == 0:
            raise ValueError("must use at least one of 'use_osm_counts', 'use_osm_areas' and 'use_osm_lengths'")


        if self.osm_tags_indexes is None:
            self.osm_tags_indexes = np.arange(self.osm_number_of_tags)

        self.input_dim = (use_osm_counts + use_osm_areas + use_osm_lengths) * len(self.osm_tags_indexes)


        layers = [
                nn.Linear(self.input_dim, layers_spec[0]),
                activation_from_str(activation_fn)
        ]
        
        for i in range(len(layers_spec)-1):
            layers.append(nn.Linear(layers_spec[i], layers_spec[i+1]))
            layers.append(activation_from_str(activation_fn))

        # last layer
        layers.append(nn.Linear(layers_spec[-1], output_dim))

        if output_activation_fn is not None:
            layers.append(activation_from_str(output_activation_fn))

        
        self.layers = nn.Sequential(*layers)
        
    def make_input(self, batch):
        x = []
        if self.use_osm_counts:
            x.append(batch['osm_ohecount'][:, self.osm_tags_indexes])
        if self.use_osm_areas:
            x.append(batch['osm_ohearea'][:, self.osm_tags_indexes])
        if self.use_osm_lengths:
            x.append(batch['osm_ohelength'][:, self.osm_tags_indexes])

        x = torch.cat(x, axis=1).type(torch.float)
        return x

    def forward(self, batch):

        x = self.make_input(batch)

        x = self.layers(x)     

        return x
        
