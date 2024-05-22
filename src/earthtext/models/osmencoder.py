import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import activation_from_str
from ..osm.multilabel import OSMCodeSets

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
                 use_osm_counts = True,
                 use_osm_areas = True,
                 use_osm_lengths = True,
                 osm_codeset = 'sentinel2'
        ):
        super().__init__()

        self.output_dim = output_dim
        self.layers_spec = layers_spec
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn
        self.osm_tags_indexes = np.r_[osm_tags_indexes] if osm_tags_indexes is not None else None
        self.use_osm_counts = use_osm_counts
        self.use_osm_areas = use_osm_areas
        self.use_osm_lengths = use_osm_lengths
        self.osm_codeset = osm_codeset

        if self.use_osm_counts + self.use_osm_areas + self.use_osm_lengths == 0:
            raise ValueError("must use at least one of 'use_osm_counts', 'use_osm_areas' and 'use_osm_lengths'")


        kvmerged = OSMCodeSets.get(osm_codeset)['kvmerged']

        if self.osm_tags_indexes is None:
            self.osm_tags_indexes = np.arange(len(kvmerged.inverse_codes))

        self.osm_tags_names = np.r_[[kvmerged.inverse_codes[i] for i in self.osm_tags_indexes]]

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
        

    def get_tag_names_for_osmcounts(self, osmcounts, min_count=1):
        """
        gets the tag names for the osmcounts>min_count taking into
        account the classes selected in osm_tag_indexes
        """
        # in case we are given a full unfiltered vector
        if len(osmcounts)==99:
            osmcounts = osmcounts[self.osm_tags_indexes]

        r = np.r_[self.osm_tags_names][osmcounts.astype(int)>=min_count]
        return r
        

    def make_input(self, batch):
        """
        assembles counts, areas and lengths into a single vector
        filtering by 'self.osm_tag_indexes' and converting to 
        tensor if required.
        """
        x = []
        mtype = self.layers[0].weight.type()
        mdevice = self.layers[0].weight.device
        if self.use_osm_counts:
            xi = batch['osm_ohecount'][:, self.osm_tags_indexes]
            if isinstance(xi, np.ndarray):
                xi = torch.tensor(xi, device=mdevice).type(mtype)
            x.append(xi)
        if self.use_osm_areas:
            xi = batch['osm_ohearea'][:, self.osm_tags_indexes]
            if isinstance(xi, np.ndarray):
                xi = torch.tensor(xi, device=mdevice).type(mtype)
            x.append(xi)
        if self.use_osm_lengths:
            xi = batch['osm_ohelength'][:, self.osm_tags_indexes]
            if isinstance(xi, np.ndarray):
                xi = torch.tensor(xi, device=mdevice).type(mtype)
            x.append(xi)

        x = torch.cat(x, axis=1).type(torch.float)
        return x

    def unmake_input(self, x):
        
        n = len(self.osm_tags_indexes)

        nx = x.shape[1]//3
        if n*3 != x.shape[1]:
            raise ValueError(f"osm vector size is not divisible by 3")

        if nx != n:
            raise ValueError(f"osm vector size {nx} does not match the number of restricted labels in 'osm_tag_indexes' which is {n}")


        return {'osm_ohecount': x[:n],
                'osm_ohearea': x[n:n*2],
                'osm_ohelength': x[n*2:]}

    def forward(self, x):

        # if we are passed a batch with separated counts, areas and lengths
        # otherwise we assume the input is already assembled
        if isinstance(x, dict):
            x = self.make_input(x)

        x = self.layers(x)     

        return x
        
