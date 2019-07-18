import logging
import json
import pickle
from operator import itemgetter
import warnings
from typing import OrderedDict, Tuple, List, Union

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from gensim.summarization.bm25 import BM25
from torchtext.data import TabularDataset

import ncn.core
from ncn.core import BaseData, Intlike, Stringlike, PathOrStr, DEVICE
from ncn.model import NeuralCitationNetwork

logger = logging.getLogger("neural_citation.evaluation")


# TODO: Document this
class Evaluator:
    """
    Evaluator class for the neural citation network. Uses a trained NCN model and BM-25 to perform
    evaluation tasks on the test set or inference on the full dataset. 
    
    ## Parameters:  
    
    - **path_to_weights** *(PathOrStr)*: Path to the weights of a pretrained NCN model. 
    - **data** *(BaseData)*: BaseData container holding train, valid, and test data.
        Also holds initialized context, title and author fields.  
    - **evaluate** *(bool=True)*: Determines the size of the BM-25 corpus used.
        If True, only the test samples will be used (model evaluation mode).
        If False, the corpus is built from the complete dataset (inference mode).   
    """
    def __init__(self, path_to_weights: PathOrStr, data: BaseData, evaluate: bool = True):
        self.data = data
        self.context, self.title, self.authors = self.data.cntxt, self.data.ttl, self.data.aut
        pad = self.title.vocab.stoi['<pad>']
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad, reduction="none")

        # instantiating model like this is bad, pass as params?
        self.model = NeuralCitationNetwork(context_filters=[4,4,5], context_vocab_size=len(self.context.vocab),
                                authors=True, author_filters=[1,2], author_vocab_size=len(self.authors.vocab),
                                title_vocab_size=len(self.title.vocab), pad_idx=pad, num_layers=2)
        self.model.to(DEVICE)
        self.model.load_state_dict(torch.load(path_to_weights, map_location=DEVICE))
        self.model.eval()
        logger.info(self.model.settings)

        self.eval = evaluate

        # instantiate examples, corpus and bm25 depending on mode
        logger.info(f"Creating corpus in eval={self.evaluate} mode.")
        if self.eval:
            self.examples = data.test.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = list(set([tuple(example.title_cited) for example in self.examples]))
            self.bm25 = BM25(self.corpus)
        else:
            self.examples = data.train.examples + data.train.examples+ data.train.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = list(set([tuple(example.title_cited) for example in self.examples]))
            self.bm25 = BM25(self.corpus)
            
            # load mapping to give proper recommendations
            with open("assets/title_tokenized_to_full.json", "rb") as fp:
                self.title_to_full = json.load(fp)

        # load mapping dictionaries for inference
        with open("assets/context_to_cited_indices.pkl", "rb") as fp:
            self.context_cited_indices = pickle.load(fp)
        with open("assets/title_to_aut_cited.pkl", "rb") as fp:
            self.title_aut_cited = pickle.load(fp)


    def _get_bm_top(self, query: List[str]) -> List[List[str]]:
        """
        Uses BM-25 to compute the most similar titles in the corpus given a query. 
        The query can either be passed as string or a list of strings (tokenized string). 
        Returns the tokenized most similar corpus titles.
        A maximum number of 2048 titles is returned. Only titles with similarity values > 0 are returned.  

        ## Parameters:  
    
        - **query** *(Stringlike)*: Query in string or tokenized form. 

        ## Output:  
        
        - **indices** *(List[int])*: List of corpus indices with the highest similary rating to the query.   
        """
        # sort titles according to score and return indices
        scores = [(score, title) for score, title in zip(self.bm25.get_scores(query), self.corpus)]
        scores = sorted(scores, key=itemgetter(0), reverse=True)
        # TODO: Reset to top 2048 again
        try:
            return [title for score, title in scores if score > 0][:48]
        except IndexError:
            return [title for score, title in scores if score > 0]


    def recall(self, x: Intlike) -> Union[float, List[float]]:
        """
        Computes recall @x metric on the test set for model evaluation purposes.  
        
        ## Parameters:  
        * *(shapes)
        - **x** *(Intlike)*: Specifies at which level the recall is computed. 
            If a list is passed, the recall is calculated for each in in the list and returned as list of floats. 
        
        ## Output:  
        
        - **recall** *(Union[float, List[float]])*: Float or list of floats with recall @x value.    
        """
        if not self.eval: warnings.warn("Performing evaluation on all data. This hurts performance.", RuntimeWarning)
        
        scores = []
        if isinstance(x, int):
            for example in self.data.test:
                # numericalize query
                context = self.context.numericalize([example.context])
                citing = self.context.numericalize([example.authors_citing])
                context = context.to(DEVICE)
                citing = citing.to(DEVICE)

                top_titles = self._get_bm_top(example.context)
                top_authors = [self.title_aut_cited[tuple(title)] for title in top_titles]
                
                # concat with true citations for context if not already selecte
                indices = self.context_cited_indices[tuple(example.context)]
                append_count = 0
                for i in indices:
                    if self.examples[i].title_cited not in top_titles: 
                        top_titles.append(self.examples[i].title_cited)
                        top_authors.append(self.examples[i].authors_cited)
                        append_count += 1

                

                logger.debug(f"Number of candidate authors {len(top_authors)}.")
                logger.debug(f"Number of candidate titles {len(top_titles)}.")
                assert len(top_authors) == len(top_titles), "Evaluation title and author lengths don't match!"

                # prepare batches
                citeds = self.authors.numericalize(self.authors.pad(top_authors))
                titles = self.title.numericalize(self.title.pad(top_titles))
                citeds = citeds.to(DEVICE)
                titles = titles.to(DEVICE)

                # repeat context and citing to len(indices) and calculate loss for single, large batch
                context = context.repeat(len(top_titles), 1)
                citing = citing.repeat(len(top_titles), 1)
                msg = "Evaluation batch sizes don't match!"
                assert context.shape[0] == citing.shape[0] == citeds.shape[0] == titles.shape[1], msg

                logger.debug(f"Context shape: {context.shape}.")
                logger.debug(f"Citing shape: {citing.shape}.")
                logger.debug(f"Titles shape: {titles.shape}.")
                logger.debug(f"Citeds shape: {citeds.shape}.")

                # calculate scores
                output = self.model(context = context, title = titles, authors_citing = citing, authors_cited = citeds)
                output = output[1:].view(-1, output.shape[-1])
                titles = titles[1:].view(-1)

                logger.debug(f"Output shape: {output.shape}")
                logger.debug(f"Titles shape: {titles.shape}")

                scores = self.criterion(output, titles)
                _, index = scores.topk(x, largest=False, sorted=True, dim=0)

                logger.debug(f"Index: {index}")
                logger.debug(f"Range of true titles: {len(top_titles) - 1} - {len(top_titles) - 1 - append_count}")

                scored = 0
                for i in append_count:
                    if len(top_titles) - (i + 1) in index: scored += 1
                
                scores.append(scored/append_count)

            return sum(scored) / len(self.data.test)

        elif isinstance(x, list):
            for at_x in x:
                scored = 0
        

    # TODO: For a query return the best citation context. Need to preprocess with context field first
    # Join top 5 titles and use dict to map back to titles
    # should have option to return attentions somehow
    def recommend(self, query: Stringlike, citing: Stringlike, top_x: int = 5):
        if self.eval: warnings.warn("Performing inference only on the test set.", RuntimeWarning)
        
        if isinstance(query, str): 
            query = self.context.tokenize(query)
        if isinstance(citing, str):
            citing = self.authors.tokenize(citing)

         
        with torch.no_grad():
            top_titles = self._get_bm_top(query)
            top_authors = [self.title_aut_cited[tuple(title)] for title in top_titles]
            assert len(top_authors) == len(top_titles), "Evaluation title and author lengths don't match!"

            context = self.context.numericalize([query])
            citing = self.context.numericalize([citing])
            context = context.to(DEVICE)
            citing = citing.to(DEVICE)

            # prepare batches
            citeds = self.authors.numericalize(self.authors.pad(top_authors))
            titles = self.title.numericalize(self.title.pad(top_titles))
            citeds = citeds.to(DEVICE)
            titles = titles.to(DEVICE)

            logger.debug(f"Evaluation title shapes: {titles.shape}")
            logger.debug(f"Batch type: {titles.type()}")

            # repeat context and citing to len(indices) and calculate loss for single, large batch
            context = context.repeat(len(top_titles), 1)
            citing = citing.repeat(len(top_titles), 1)
            msg = "Evaluation batch sizes don't match!"
            assert context.shape[0] == citing.shape[0] == citeds.shape[0] == titles.shape[1], msg

            # calculate scores
            output = self.model(context = context, title = titles, authors_citing = citing, authors_cited = citeds)
            output = output[1:].permute(1,2,0)
            titles = titles[1:].permute(1,0)

            logger.debug(f"Evaluation output shapes: {output.shape}")
            logger.debug(f"Evaluation title shapes: {titles.shape}")

            scores = self.criterion(output, titles)
            scores = torch.einsum("ij -> i", scores)
            logger.debug(f"Evaluation scores shape: {scores.shape}")
            _, index = scores.topk(top_x, largest=False, sorted=True, dim=0)

            recommended = [" ".join(top_titles[i]) for i in index]
        
        return pd.DataFrame({f"top_{i+1}": self.title_to_full[title] for i, title in enumerate(recommended)},
                             index=["Recommended Title"]).transpose()

        
        



        
