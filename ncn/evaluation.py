import logging
import json
import pickle
from operator import itemgetter
import warnings
from typing import Dict, Tuple, List, Union

import torch
from torch import nn
import torch.nn.functional as F
from gensim.summarization.bm25 import BM25
from torchtext.data import TabularDataset
from tqdm import tqdm_notebook

import ncn.core
from ncn.core import BaseData, Stringlike, PathOrStr, DEVICE, Filters
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
    def __init__(self, context_filters: Filters, author_filters: Filters,
                 num_filters: int, embed_size:int,
                 path_to_weights: PathOrStr, data: BaseData, 
                 evaluate: bool = True, show_attention: bool = False):
        self.data = data
        self.context, self.title, self.authors = self.data.cntxt, self.data.ttl, self.data.aut
        self.pad = self.title.vocab.stoi['<pad>']
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.pad, reduction="none")

        self.model = NeuralCitationNetwork(context_filters=context_filters,
                                            author_filters=author_filters,
                                            context_vocab_size=len(self.context.vocab),
                                            title_vocab_size=len(self.title.vocab),
                                            author_vocab_size=len(self.authors.vocab),
                                            pad_idx=self.pad,
                                            num_filters=num_filters,
                                            authors=True, 
                                            embed_size=embed_size,
                                            num_layers=2,
                                            hidden_size=num_filters,
                                            dropout_p=0.2,
                                            show_attention=False)
        self.model.to(DEVICE)
        self.model.load_state_dict(torch.load(path_to_weights, map_location=DEVICE))
        self.model.eval()
        logger.info(self.model.settings)

        self.eval = evaluate
        self.show_attention = show_attention

        # instantiate examples, corpus and bm25 depending on mode
        logger.info(f"Creating corpus in eval={self.eval} mode.")
        if self.eval:
            self.examples = data.test.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = list(set([tuple(example.title_cited) for example in self.examples]))
            self.bm25 = BM25(self.corpus)
            self.context_cited_indices = self._get_context_title_indices(self.examples)
        else:
            self.examples = data.train.examples + data.train.examples+ data.train.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = list(set([tuple(example.title_cited) for example in self.examples]))
            self.bm25 = BM25(self.corpus)
            
            # load mapping to give proper recommendations
            with open("assets/title_tokenized_to_full.pkl", "rb") as fp:
                self.title_to_full = pickle.load(fp)

        
        with open("assets/title_to_aut_cited.pkl", "rb") as fp:
            self.title_aut_cited = pickle.load(fp)

    @staticmethod
    def _get_context_title_indices(examples: List) -> Dict[Tuple[str, ...], List[str]]:
        mapping = {}
        for i, example in enumerate(examples):
            key = tuple(example.context)
            if key not in mapping.keys():
                mapping[key] = [i]
            else:
                mapping[key].append(i)
        
        return mapping

    def _get_bm_top(self, query: List[str]) -> List[List[str]]:
        """
        Uses BM-25 to compute the most similar titles in the corpus given a query. 
        The query can either be passed as string or a list of strings (tokenized string). 
        Returns the tokenized most similar corpus titles.
        Only titles with similarity values > 0 are returned.
        A maximum number of 2048 titles is returned in eval mode. 
        For recommendations, the top 256 titles are returned.  

        ## Parameters:  
    
        - **query** *(Stringlike)*: Query in string or tokenized form. 

        ## Output:  
        
        - **indices** *(List[int])*: List of corpus indices with the highest similary rating to the query.   
        """
        # sort titles according to score and return indices
        scores = [(score, title) for score, title in zip(self.bm25.get_scores(query), self.corpus)]
        scores = sorted(scores, key=itemgetter(0), reverse=True)

        # Return top 2048 for evaluation purpose, cut to half for recommendations to prevent memory errors
        if self.eval:
            try:
                return [title for score, title in scores][:256]
            except IndexError:
                return [title for score, title in scores]
        else:
            try:
                return [title for score, title in scores if score > 0][:1028]
            except IndexError:
                return [title for score, title in scores if score > 0]


    def recall(self, x: int) -> Union[float, List[float]]:
        """
        Computes recall @x metric on the test set for model evaluation purposes.  
        
        ## Parameters:  
        * *(shapes)
        - **x** *(int)*: Specifies at which level the recall is computed.  
        
        ## Output:  
        
        - **recall** *(Union[float, List[float]])*: Float or list of floats with recall @x value.    
        """
        if not self.eval: warnings.warn("Performing evaluation on all data. This hurts performance.", RuntimeWarning)
        
        recall_list = []
        with torch.no_grad():
            for example in tqdm_notebook(self.data.test[:20000]):
                # numericalize query
                context = self.context.numericalize([example.context])
                citing = self.context.numericalize([example.authors_citing])
                context = context.to(DEVICE)
                citing = citing.to(DEVICE)

                # catch contexts and citings shorter than filter lengths and pad manually
                if context.shape[1] < 5:
                    assert self.pad == 1, "Padding index doesn't match tensor index!"
                    padded = torch.ones((1,5), dtype=torch.long)
                    padded[:, :context.shape[1]] = context
                    context = padded.to(DEVICE)
                if citing.shape[1] < 2:
                    padded = torch.ones((1,2), dtype=torch.long)
                    padded[:, :citing.shape[1]] = citing
                    citing = padded.to(DEVICE)

                top_titles = self._get_bm_top(example.context)
                logger.info(f"True title in top_titles: {example.title_cited in top_titles}.")
                top_authors = [self.title_aut_cited[tuple(title)] for title in top_titles]
                
                # add all true cited titles (can be multiple per context)
                indices = self.context_cited_indices[tuple(example.context)]
                append_count = 0
                for i in indices: 
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
                output = output[1:].permute(1,2,0)
                titles = titles[1:].permute(1,0)

                logger.debug(f"Evaluation output shapes: {output.shape}")
                logger.debug(f"Evaluation title shapes: {titles.shape}")

                scores = self.criterion(output, titles)
                scores = scores.sum(dim=1)
                logger.info(f"Evaluation scores shape: {scores.shape}")

                _, index = scores.topk(x, largest=False, sorted=True, dim=0)

                logger.info(f"Index: {index}")
                logger.info(f"Lowest scores: {scores[index]}")
                logger.info(f"Range of true titles: {len(top_titles) - 1} - {len(top_titles) - 1 - append_count}")

                # check how many of the concatenated (=true) titles have been returned
                scored = 0
                for i in range(append_count):
                    if len(top_titles) - (i + 1) in index: scored += 1
                    
                logger.info(f"Scored {scored} out of {append_count} titles at {x}.")
                
                recall_list.append(scored/append_count)

            return sum(recall_list) / len(self.data.test[:20000])
        
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

            # repeat context and citing to len(indices) and calculate loss for single, large batch
            context = context.repeat(len(top_titles), 1)
            citing = citing.repeat(len(top_titles), 1)
            msg = "Evaluation batch sizes don't match!"
            assert context.shape[0] == citing.shape[0] == citeds.shape[0] == titles.shape[1], msg

            # calculate scores
            if self.show_attention:
                output, attention = self.model(context = context, title = titles, 
                                               authors_citing = citing, authors_cited = citeds)
            else:
                output = self.model(context = context, title = titles, 
                                    authors_citing = citing, authors_cited = citeds)
            output = output[1:].permute(1,2,0)
            titles = titles[1:].permute(1,0)

            logger.debug(f"Evaluation output shapes: {output.shape}")
            logger.debug(f"Evaluation title shapes: {titles.shape}")

            scores = self.criterion(output, titles)
            scores = scores.sum(dim=1)
            logger.debug(f"Evaluation scores shape: {scores.shape}")
            _, index = scores.topk(top_x, largest=False, sorted=True, dim=0)

            recommended = [" ".join(top_titles[i]) for i in index]
        
        if self.show_attention:
            return {i: self.title_to_full[title] for i, title in enumerate(recommended)}, attention[:, index, :]

        return {i: self.title_to_full[title] for i, title in enumerate(recommended)}

        