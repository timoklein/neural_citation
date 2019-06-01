from pathlib import Path
import re
from ast import literal_eval
from typing import Union
import spacy

PathOrStr = Union[Path, str]

"""
1. Step of preprocessing:
    For each document with all available data create
        Dictionary/JSON of the form:
        {
            "title_citing": "...",
            "authors_citing": "...",
            "citations": [
                {
                    "context": "...",
                    "title_cited": "...",
                    "authors_cited": "...",
                }
            ]
        }

2. Step of preprocessing:
    For data JSON:
        1. Tokenize
        2. Lemmatize
        3. Remove formulas
        4. Prune vocabulary?

3. Step of preprocessing:
    Apply GloVe embeddings and store results as torch tensors of the form 1xembed_dimxseq_len

4. Get in consumption format for NCN
"""