import torch
import spacy
from spacy.lang.en import English
from typing import Union, List
from pathlib import Path
from typing import NamedTuple
from torchtext.data import Field, BucketIterator

PAD = 0
BOS = 1
EOS = 2
UNK = 3

PathOrStr = Union[Path, str]
"""Custom type for Paths or pathlike objects."""

Filters = List[int]
"""Custom data type representing a list of filter lengths."""

CITATION_PATTERNS = r"<DBLP:.*?>|<GC:.*?>"
"""Regex patterns for matching citations in document sentences."""

STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
"""Set of stopwords obtained via spacy."""

MAX_TITLE_LENGTH = 40
"""Maximum decoder sequence length. Determines the number of attention weights."""

MAX_CONTEXT_LENGTH = 100
"""Maximum encoder sequence length."""

MAX_AUTHORS = 7
"""Maximum number of authors considered"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Check for a GPU globally."""

class TrainingData(NamedTuple):
    """
    Container holding the data needed to train the NCN model. 
    
    ## Attributes:  
    
    - **cntxt** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for context data.  
    - **ttl** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for title data.  
    - **aut** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for author data.  
    - **train_iter, vailid_iter, test_iter** *(torch.text.data.BucketIterator)*:  
        Iterators containing the training examples of the form context, citing_authors, (title, title_length), cited_authors.
        Data is bucketted according to the title length.        
    """
    cntxt: Field
    ttl: Field
    aut: Field
    train_iter: BucketIterator
    valid_iter: BucketIterator
    test_iter: BucketIterator

