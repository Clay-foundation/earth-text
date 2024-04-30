from .. import datamodules
from ..osm.multilabel import kvmerged
from ..datamodules.components import chipmultilabel
from loguru import logger
import os
from omegaconf import OmegaConf
import hydra
import torch
from progressbar import progressbar as pbar
import numpy as np
from loguru import logger

def similarity(t1,t2):
    
    t1 = t1 / np.linalg.norm(t1)
    t2 = t2 / np.linalg.norm(t2)
    return t1.dot(t2) 

def get_similarity_sorted_indexes(vectorbase, query, topn=None):
    if topn is not None and (not isinstance(topn, int) or topn<=0):
        raise ValueError(f"topn must be a positive integer, but got {topn}")
    ntargets = (vectorbase.T / np.linalg.norm(vectorbase,axis=1)).T
    nquery   = query / np.linalg.norm(query)
    sims = ntargets.dot(nquery)
    sorted_idxs = np.argsort(sims)[::-1]
    sorted_sims = sims[sorted_idxs]

    if topn is not None:
        sorted_idxs = sorted_idxs[:topn]
        sorted_sims = sorted_sims[:topn]

    return sorted_idxs, sorted_sims


def get_embeddings_osmvectors_predictions(dataloader, model=None):

    dataloader.dataset.get_chip_id = True

    embeddings = []
    osmvectors = {'osm_ohecount':[] , 'osm_ohearea': [], 'osm_ohelength': []}
    predicted_embeddings = []
    chip_ids = []

    for batch in pbar(dataloader):
        t = batch['embedding']

        if model is not None:
            o = model(batch)
            for oi in o.detach().numpy():
                predicted_embeddings.append(oi)

        for k in osmvectors.keys():
            for xi in batch[k].detach().numpy():
                osmvectors[k].append(xi)
    
        for ti in t.detach().numpy():
            embeddings.append(ti)    
        
        for ci in batch['chip_id']:
            chip_ids.append(ci)

    embeddings = np.r_[embeddings]
    for k in list(osmvectors.keys()):
        osmvectors[k] = np.r_[osmvectors[k]]


    if model is not None:
        predicted_embeddings = np.r_[predicted_embeddings]
        return chip_ids, embeddings, osmvectors, predicted_embeddings
    else:
        return chip_ids, embeddings, osmvectors

class OSMClayModelSearcher:

    def __init__(self, model_ckpt, search_target = 'train'):
        logger.info(f"search target is '{search_target}'")
        self.model_ckpt_fname = model_ckpt
        self.model_conf_fname = model_ckpt[:-5] + ".yaml"
        self.search_target = search_target
        self.conf = OmegaConf.load(self.model_conf_fname)

        if not os.path.isfile(self.model_ckpt_fname) or not os.path.isfile(self.model_conf_fname):
            raise ValueError("cannot find model or conf")

    def get_dataloader(self, search_target=None):
        if search_target is None:
            search_target = self.search_target

        if search_target == 'train':
            search_dataloader = self.dataloader.train_dataloader(shuffle=False)
        elif search_target == 'test':
            search_dataloader = self.dataloader.test_dataloader(shuffle=False)
        elif search_target == 'val':
            search_dataloader = self.dataloader.val_dataloader(shuffle=False)

        search_dataloader.dataset.get_chip_id = True
        return search_dataloader

    def get_search_dataset(self, search_target=None):
        if search_target is None:
            search_target = self.search_target

        if search_target == 'train':
            r = self.dataloader.train_dataset
        elif search_target == 'test':
            r = self.dataloader.test_dataset
        elif search_target == 'val':
            r = self.dataloader.val_dataset

        r.get_chip_id = True
        return r


    def setup(self):
        logger.info("loading model")
        self.model = hydra.utils.instantiate(self.conf.model)
        self.model.load_state_dict(torch.load(self.model_ckpt_fname))

        logger.info("initializing dataloaders")
        self.dataloader = hydra.utils.instantiate(self.conf.dataloader)

        logger.info("loading embeddings and predictions")
        self.dataloader.disable_chip_loading()
        search_dataloader = self.get_dataloader()
        chip_ids,embeddings,osmvectors = get_embeddings_osmvectors_predictions(search_dataloader)

        self.searchdb = {    
            'chip_ids': chip_ids,
            'normalized_osm_vectors': osmvectors,
            'embeddings': embeddings,
            'original_osm_vectors': chipmultilabel.unnormalize_osm_vector(osmvectors)
        }

    def predict_embeddings(self, normalized_query_vector):
        return self.model(normalized_query_vector).detach()

    def make_search(self, predicted_embedding, topn=None):
        idxs, scores = get_similarity_sorted_indexes(self.searchdb['embeddings'], predicted_embedding, topn=topn)
        search_result = {k:({kk:vv[idxs] for kk,vv in v.items()} if isinstance(v, dict) else np.r_[v][idxs]) for k,v in self.searchdb.items()}
        
        search_result['idxs'] = idxs
        search_result['scores'] = scores
        return search_result

    def get_tagnames(self, original_query_vector, search_result, min_count=1):
        query_tags = self.model.get_tag_names_for_osmcounts(original_query_vector)
        search_result['tags'] = [list(self.model.get_tag_names_for_osmcounts(i)) for i in search_result['original_osm_vectors']['osm_ohecount']]
        
        return query_tags, search_result


class QueryAutocompletionSampler:

    def __init__(self, searcher, query_source = 'test'):
        """
        searcher: searcher objects
        query_dataloader: from where queries are sampled
        """
        logger.info(f"search target is {searcher.search_target}, query source is {query_source}")
        self.searcher = searcher
        self.query_source = query_source
        self.query_dataloader = searcher.get_dataloader(query_source)
        logger.info("loading all samples from query source")
        self.q_chip_ids,\
        self.q_embeddings,\
        self.q_normalized_osmvectors = get_embeddings_osmvectors_predictions(self.query_dataloader)
        self.q_original_osmvectors = chipmultilabel.unnormalize_osm_vector(self.q_normalized_osmvectors)

    def sample_queries(self, n_queries):
        o = self.q_normalized_osmvectors 
        idxs = np.random.permutation(len(o['osm_ohecount']))[:n_queries]
        normalized_query_vector = {k: v[idxs] for k,v in o.items()}
        original_query_vector   = {k: v[idxs] for k,v in self.q_original_osmvectors.items()}

        return {'normalized_query_vector': normalized_query_vector,
                'original_query_vector': original_query_vector,
                'indexes': idxs
                }

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

            