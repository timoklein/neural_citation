import logging
from operator import itemgetter
import warnings
from typing import OrderedDict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from gensim.summarization.bm25 import BM25
from torchtext.data import TabularDataset

import ncn.core
from ncn.core import BaseData, Ints

logger = logging.getLogger("neural_citation.inference")


# TODO: Document this
class Evaluator:
    def __init__(self, model: nn.Module, data: BaseData, pad: int = 1, eval: bool = True):
        self.model = model
        self.data = data
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad, reduction="sum")

        # instantiate corpus and bm25 depending on mode
        if eval:
            self.corpus = [example.title_cited for example in data.test.examples]
            self.bm25 = BM25(corpus)
        else:
            examples = data.train.examples + data.train.examples+ data.train.examples
            self.corpus = [example.title_cited for example in examples]
            self.bm25 = BM25(corpus)

    def _get_bm_top(self, query: str) -> List[Tuple[float, str]]:
        q = data.cntxt.tokenize(query)

        # create {bm_score, title} dict and sort according to score
        scores = [
            (scores[0], scores[1]) for i, scores in enumerate(zip(bm25.get_scores(q), self.corpus))
            if bm25.get_score(q, i) > 0
        ]
        scores = sorted(scores, key=itemgetter(0), reverse=True)
        try:
            return scores[:2048]
        except IndexError:
            return scores

    # TODO: get top 2048, pass through ncn for a single context, rerank according to ncn scores
    # Then evaluate if rerank is in top x, compute and return total score
    # Check if it's single int or list of ints and act accordingly
    def recall(self, x: Ints):
        if not eval: warnings.warn("Performing evaluation on training data. This is bad practice.", RuntimeWarning)
        pass
        

    # TODO: For a query return the best citation context. Need to preprocess with context field first
    def recommend(self, query: str):
        if eval: warnings.warn("Performing inference only on the test set.", RuntimeWarning)
        pass