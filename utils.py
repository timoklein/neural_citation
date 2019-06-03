from pathlib import Path
import re
from ast import literal_eval
from typing import Union, Collection
import spacy

PathOrStr = Union[Path, str]

"""
1. Step of preprocessing:
    For each context with all available data create
        Dictionary/JSON of the form:
        {   "context": Tensor w. shapes 300xlen(context),
            "title_citing": Tensor w. shapes 300xlen(title_citing),
            "authors_citing": Vector with length author_vocab,     
            "title_cited": "...",
            "authors_cited": "...",
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

def process_text(text: str, delimiter: str = "\n============\n") -> Collection:
    text = re.sub("<formula>", '')
    sentences = text.split(delimiter)
    contexts = []
    for sentence in sentences:
        if re.search("<DBLP:|<GC:", sentence):
            contexts.append(sentence)


def process_refs(refs: str, delimiter: str = "\n") -> Collection:
    return refs.split(delimiter)


def prepare_data(path: PathOrStr) -> None:
    """
    Prepare the arxiv CS dataset and save in JSON format.
    INPUTS:
    * __path__(PathOrStr):         Path or string to files
    """
    path = Path(path)
    for textpath in path.glob("*.txt"):
        metapath = textpath.with_suffix(".meta")
        refpath = textpath.with_suffix(".refs")

        with open(textpath, 'r') as f:
            text = f.read()
        with open(metapath, 'r') as f:
            meta = f.read()
        with open(refpath, 'r') as f:
            refs = f.read()
        
        # throw away incomplete data instances before further processing rest
        if len(text) == 0 or len(meta) == 0 or len(refs) == 0:
            return
        else:
            meta = literal_eval(meta)
            text = process_text(text)
            refs = process_refs(refs)