import logging
from operator import itemgetter
import warnings
from typing import OrderedDict, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from gensim.summarization.bm25 import BM25
from torchtext.data import TabularDataset

import ncn.core
from ncn.core import BaseData, Ints, PathOrStr, DEVICE
from ncn.model import NeuralCitationNetwork

logger = logging.getLogger("neural_citation.inference")


# TODO: Document this
class Evaluator:
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
        logger.info(model.settings)

        # instantiate examples, corpus and bm25 depending on mode
        logger.info(f"Creating corpus in eval={eval} mode.")
        if eval:
            self.examples = data.test.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = [example.title_cited for example in self.examples]
            self.bm25 = BM25(corpus)
        else:
            self.examples = data.train.examples + data.train.examples+ data.train.examples
            logger.info(f"Number of samples in BM25 corpus: {len(self.examples)}")
            self.corpus = [example.title_cited for example in self.examples]
            self.bm25 = BM25(corpus)

    def _get_bm_top(self, query: str) -> List[Tuple[float, str]]:
        q = self.context.tokenize(query)

        # sort titles according to score and return indices
        scores = [
            (score, index) for score, index in enumerate(bm25.get_scores(q))
            if bm25.get_score(q, index) > 0
        ]
        scores = sorted(scores, key=itemgetter(0), reverse=True)
        try:
            return [index for _, index in scores][:2048]
        except IndexError:
            return [index for _, index in scores]


    def recall(self, x: Ints):
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
                candidate_authors = [self.examples[i].authors_cited for i in indices] + example.authors_cited
                candidate_titles = [self.examples[i].title_cited for i in indices] + example.title_cited
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
                output = model(context = context, title = titles, authors_citing = citing, authors_cited = citeds)
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
            
            return scored / len(self.data.test)

        elif isinstance(x, list):
            scores = []
            for at_x in x:
                scored = 0
        

    # TODO: For a query return the best citation context. Need to preprocess with context field first
    def recommend(self, query: str, top_x: int = 5):
        if eval: warnings.warn("Performing inference only on the test set.", RuntimeWarning)
        q = self.data.cntxt.tokenize(query)
        # get indices
        # return top x