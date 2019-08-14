import torch
import spacy
import nltk
import logging
from spacy.lang.en import English
from typing import Union, List
from pathlib import Path
from typing import NamedTuple, Set
from torchtext.data import Field, BucketIterator, TabularDataset


# Custom data types and structures
PathOrStr = Union[Path, str]
"""Custom type for Paths or pathlike objects."""

Filters = List[int]
"""Custom data type representing a list of filter lengths."""

Stringlike = Union[str, List[str]]
"""Single string or list of strings for evaluating recall."""

class IteratorData(NamedTuple):
    """ Container holding the iterators needed to train the NCN model."""

    cntxt: Field
    """**cntxt** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for context data."""
    ttl: Field
    """**ttl** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for title data."""
    aut: Field
    """**aut** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for author data."""
    train_iter: BucketIterator
    """
    **train_iter** *(torch.text.data.BucketIterator)*:  
    Iterator containing the training samples of the form context, citing_authors, title, cited_authors.
    Data is bucketted according to the title length.
    """
    valid_iter: BucketIterator
    """
    **valid_iter** *(torch.text.data.BucketIterator)*:  
    Iterator containing samples for the validation pass. Format: context, citing_authors, title, cited_authors.
    Data is bucketted according to the title length.
    """
    test_iter: BucketIterator
    """
    **test_iter** *(torch.text.data.BucketIterator)*:  
    Iterator containing samples for the test pass. Format: context, citing_authors, title, cited_authors.
    Data is bucketted according to the title length.
    """


class BaseData(NamedTuple):
    """Container holding base data for the arxiv CS dataset."""

    cntxt: Field
    """**cntxt** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for context data"""
    ttl: Field
    """**ttl** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for title data."""
    aut: Field
    """**aut** *(torch.text.data.Field)*: Field containing preprocessing steps and vocabulary for author data."""
    train: TabularDataset
    """
    **train** *(torch.text.data.TabularDataset)*:  
    Dataset containing the training samples of the form context, citing_authors, title, cited_authors.
    """
    valid: TabularDataset
    """
    **valid** *(torch.text.data.TabularDataset)*:  
    Dataset containing the validation samples of the form context, citing_authors, title, cited_authors.
    """
    test: TabularDataset
    """
    **test** *(torch.text.data.TabularDataset)*:  
    Dataset containing the training samples of the form context, citing_authors, title, cited_authors.
    """


# Global constants
CITATION_PATTERNS = r"<DBLP:.*?>|<GC:.*?>"
"""Regex patterns for matching citations in document sentences."""


MAX_TITLE_LENGTH = 15
"""Maximum decoder sequence length. Also determines the number of attention weights."""

MAX_CONTEXT_LENGTH = 35
"""Maximum encoder sequence length."""

MAX_AUTHORS = 5
"""Maximum number of authors considered"""

SEED = 34
"""RNG seed for reproducability."""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Check for a GPU globally."""


# base logger for the ncn module
logging.basicConfig(level=logging.INFO, style='$')
logger = logging.getLogger(__name__)
"""
    Base logger for the neural citation package.
    The package wide logging level is set here.
"""

# general functions
def get_stopwords() -> Set:
    """
    Returns spacy and nltk stopwords unified into a single set.   
    
    ## Output:  
    
    - **STOPWORDS** *(Set)*: Set containing the stopwords for preprocessing 
    """
    STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
    nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
    STOPWORDS.update(nltk_stopwords)
    return STOPWORDS
    