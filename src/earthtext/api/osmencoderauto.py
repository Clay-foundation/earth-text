from loguru import logger
import numpy as np
import os
from earthtext.osm.multilabel import kvmerged
from omegaconf import DictConfig, OmegaConf
import hydra
from progressbar import progressbar as pbar
import torch

def get_osmvectors(dataloader):

    dataloader.dataset.get_chip_id = True

    osmvectors = {'osm_ohecount':[] , 'osm_ohearea': [], 'osm_ohelength': []}
    chip_ids = []

    for batch in pbar(dataloader):
        for k in osmvectors.keys():
            for xi in batch[k].detach().numpy():
                osmvectors[k].append(xi)

    for k in list(osmvectors.keys()):
        osmvectors[k] = np.r_[osmvectors[k]]

    return chip_ids, osmvectors
    
class OSMEncoderWithAutocompletion:

    def __init__(self, model_ckpt, autocompletion_source = 'train'):
        logger.info(f"osmencoder model is {model_ckpt}")
        logger.info(f"autocompletion source is '{autocompletion_source}'")
        self.model_ckpt_fname = model_ckpt
        self.model_conf_fname = model_ckpt[:-5] + ".yaml"
        self.autocompletion_source = autocompletion_source
        self.conf = OmegaConf.load(self.model_conf_fname)

        if not os.path.isfile(self.model_ckpt_fname) or not os.path.isfile(self.model_conf_fname):
            raise ValueError("cannot find model or conf")

    def setup(self):
        self.conf.dataloader['embeddings_folder'] = None
        self.conf.dataloader['chips_folder'] = None
        
        logger.info("loading model")
        self.model = hydra.utils.instantiate(self.conf.model)
        self.model.load_state_dict(torch.load(self.model_ckpt_fname))
        
        logger.info("initializing dataloaders")
        self.dataloader = hydra.utils.instantiate(self.conf.dataloader)
        
        if self.autocompletion_source == 'train':
            self.source_dataloader = self.dataloader.train_dataloader()
        elif self.autocompletion_source == 'test':
            self.source_dataloader = self.dataloader.test_dataloader()
        elif self.autocompletion_source == 'val':
            self.source_dataloader = self.dataloader.val_dataloader()
        
        self.q_chipids, \
        self.q_normalized_osmvectors = get_osmvectors(self.source_dataloader)
        
        self.q_original_osmvectors = self.dataloader.train_dataset.normalizer.unnormalize_osm_vector(self.q_normalized_osmvectors)
        return self
        
    def sample_queries_with_conditions(self, min_counts={}, max_counts={}, min_areas={}, max_areas={}, n_samples=10):
        """
        for instance
            min_counts = {'building=*':100}
            max_counts = {'natural=*': 0}
        """
        query_min_counts = np.zeros(len(kvmerged.inverse_codes))
        for k,v in min_counts.items():
            query_min_counts[kvmerged.keyvals_codes[k]] = v
        
        query_max_counts = np.zeros(len(kvmerged.inverse_codes)) + np.inf
        for k,v in max_counts.items():
            query_max_counts[kvmerged.keyvals_codes[k]] = v

        query_min_areas = np.zeros(len(kvmerged.inverse_codes))
        for k,v in min_areas.items():
            query_min_areas[kvmerged.keyvals_codes[k]] = v
        
        query_max_areas = np.zeros(len(kvmerged.inverse_codes)) + np.inf
        for k,v in max_areas.items():
            query_max_areas[kvmerged.keyvals_codes[k]] = v

        filter_cmin = self.q_original_osmvectors['osm_ohecount']>=query_min_counts
        filter_cmax = self.q_original_osmvectors['osm_ohecount']<=query_max_counts

        filter_amin = self.q_original_osmvectors['osm_ohearea']>=query_min_areas
        filter_amax = self.q_original_osmvectors['osm_ohearea']<=query_max_areas

        
        compliant_indexes = np.argwhere(np.all(filter_cmin * filter_cmax * filter_amin * filter_amax, axis=1))[:,0]
        idxs = np.random.permutation(compliant_indexes)[:n_samples]
        
        normalized_query_vector = {k: v[idxs] for k,v in self.q_normalized_osmvectors.items()}
        original_query_vector   = {k: v[idxs] for k,v in self.q_original_osmvectors.items()}

        return {'normalized_query_vector': normalized_query_vector,
                'original_query_vector': original_query_vector,
                'indexes': idxs
                }

    def predict_embedding(self, min_counts, max_counts, min_areas, max_areas):
        q = self.sample_queries_with_conditions(min_counts, max_counts, min_areas, max_areas)
        query_osmvector = {k:v.mean(axis=0).reshape(1,-1) for k,v in q['normalized_query_vector'].items()}
        p = self.model(query_osmvector)[0].detach().numpy()
        return p