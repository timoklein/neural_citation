import logging
from operator import itemgetter
import warnings
from typing import OrderedDict, Tuple, List, Union

import torch
from torch import nn
import torch.nn.functional as F
from gensim.summarization.bm25 import BM25
from torchtext.data import TabularDataset

import ncn.core
from ncn.core import BaseData, Intlike, Stringlike, PathOrStr, DEVICE
from ncn.model import NeuralCitationNetwork

logger = logging.getLogger("neural_citation.inference")


# TODO: Document this
class Evaluator:
    """
    Evaluator class for the neural citation network. Uses a trained NCN model and BM-25 to perform
    evaluation tasks (recall @ x) or inference. 
    
    ## Parameters:  
    
    - **path_to_weights** *(PathOrStr)*: Path to the weights of a pretrained NCN model. 
    - **data** *(BaseData)*: BaseData container holding train, valid, and test data.
        Also holds initialized context, title and author fields.  
    - **eval** *(bool=True)*: Determines the size of the BM-25 corpus used.
        If True, only the test samples will be used (model evaluation mode).
        If False, the corpus is built from the complete dataset (inference mode).   
    """
    def __init__(self, path_to_weights: PathOrStr, data: BaseData, eval: bool = True):
        self.data = data
        self.context, self.title, self.authors = self.data.cntxt, self.data.ttl, self.data.aut
        pad = self.title.vocab.stoi['<pad>']
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad, reduction=None)

        # instantiating model like this is bad, pass as params?
        self.model = NeuralCitationNetwork(context_filters=[4,4,5], context_vocab_size=len(self.context.vocab),
                                authors=True, author_filters=[1,2], author_vocab_size=len(self.authors.vocab),
                                title_vocab_size=len(self.title.vocab), pad_idx=pad, num_layers=2)
        self.model.to(DEVICE)
        self.model.load_state_dict(torch.load(path_to_weights, map_location=DEVICE))
        self.model.eval()
        logger.info(self.model.settings)

        # instantiate examples, corpus and bm25 depending on mode
        # TODO: Remove duplicates from corpus
        logger.info(f"Creating corpus in eval={eval} mode.")
        if eval:
            self.examples = data.test.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = [example.title_cited for example in self.examples]
            self.bm25 = BM25(self.corpus)
        else:
            self.examples = data.train.examples + data.train.examples+ data.train.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = [example.title_cited for example in self.examples]
            self.bm25 = BM25(self.corpus)

    #TODO: Document
    def _get_bm_top(self, query: Stringlike) -> List[int]:
        """
        Uses BM-25 to compute the most similar titles in the corpus given a query. 
        The query can either be passed as string or a list of strings (tokenized string). 
        Returns a list of ints corresponding to the indices of the most similar corpus titles. 
        A maximum number of 2048 indices is returned. Only indices with similarity values > 0 are returned.  

        ## Parameters:  
    
        - **query** *(Stringlike)*: Query in string or tokenized form. 

        ## Output:  
        
        - **indices** *(List[int])*: List of corpus indices with the highest similary rating to the query.   
        """
        if isinstance(query, str): query = self.context.tokenize(query)

        # sort titles according to score and return indices
        scores = [
            (score, index) for index, score in enumerate(self.bm25.get_scores(query))
            if self.bm25.get_score(query, index) > 0
        ]
        scores = sorted(scores, key=itemgetter(0), reverse=True)
        try:
            return [index for _, index in scores][:2048]
        except IndexError:
            return [index for _, index in scores]

    # TODO: Document
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
        if not eval: warnings.warn("Performing evaluation on all data. This hurts performance.", RuntimeWarning)

        if isinstance(x, int):
            scored = 0
            for example in self.data.test:
                # numericalize query
                context = self.context.numericalize([example.context])
                citing = self.context.numericalize([example.authors_citing])
                context = context.to(DEVICE)
                citing = citing.to(DEVICE)

                indices = self._get_bm_top(example.context)
                # get titles, cited authors with top indices and concatenate with true citation
                candidate_authors = [self.examples[i].authors_cited for i in indices]
                candidate_authors.append(example.authors_cited)
                candidate_titles = [self.examples[i].title_cited for i in indices]
                candidate_titles.append(example.title_cited)

                logger.debug(f"Number of candidate authors {len(candidate_authors)}.")
                logger.debug(f"Number of candidate titles {len(candidate_titles)}.")
                assert len(candidate_authors) == len(candidate_titles), "Evaluation title and author lengths don't match!"

                # prepare batches
                citeds = self.authors.numericalize(self.authors.pad(candidate_authors))
                titles = self.title.numericalize(self.title.pad(candidate_titles))
                citeds = citeds.to(DEVICE)
                titles = titles.to(DEVICE)

                # repeat context and citing to len(indices) and calculate loss for single, large batch
                context = context.repeat(len(candidate_titles), 1)
                citing = citing.repeat(len(candidate_titles), 1)
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

                logger.debug(index)
                logger.debug(titles.shape[1] - 1)

                if titles.shape[1] - 1 in index: 
                    scored += 1
            
            # FIXME: Actually compute recall here
            return scored / len(self.data.test)

        elif isinstance(x, list):
            scores = []
            for at_x in x:
                scored = 0
        

    # TODO: For a query return the best citation context. Need to preprocess with context field first
    # Join top 5 titles and use dict to map back to titles
    # should have option to return attentions somehow
    def recommend(self, query: str, top_x: int = 5):
        if eval: warnings.warn("Performing inference only on the test set.", RuntimeWarning)
        q = self.data.cntxt.tokenize(query)
        # get indices
        # return top x