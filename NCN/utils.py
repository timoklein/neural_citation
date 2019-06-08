import re
import pandas as pd
import spacy
import logging
import json
from pandas import DataFrame
from pathlib import Path
from typing import Union, Collection, List, Dict


logging.basicConfig(level=logging.DEBUG, style='$')

PathOrStr = Union[Path, str]
"""Custom type for Paths or pathlike objects."""

CITATION_PATTERNS = r"<DBLP:.*?>|<GC:.*?>"
"""Regex patterns for matching citations in document sentences."""


def process_text(text: str, delimiter: str = "\\n============\\n") -> List[str]:
    """
    Preprocessing function for preprocessing arxiv CS paper text.     
    **Parameters**:   
    - *text* (str): .txt file string object containing the text of a paper.  
    - *delimiter* (str = "\\n============\\n"): token separating text sentences.        
    **Output**:  
    - List with sentences split at *delimiter*. Only sentences containing *CITATION_PATTERNS* are retained.
    """
    text = re.sub("<formula>", '', text)
    sentences = text.split(delimiter)
    contexts = []
    for sentence in sentences:
        if re.search(CITATION_PATTERNS, sentence):
            contexts.append(sentence)
    return contexts


def process_refs(refs: str, delimiter_patterns: str = "GC|DBLP") -> List[str]:
    """
    Preprocessing function for preprocessing arxiv CS paper references.     
    **Parameters**:   
    - *refs* (str): reference file string.  
    - *delimiter_patterns* (str = "GC|DBLP"): regex patterns used to split the inidividual references.     
    **Output**:  
    - List citation contexts split at *delimiter*.
    """
    refs = re.sub("\n", '', refs)
    return re.split(delimiter_patterns, refs)


# TODO: FIX ref splitting at \n to GC and DBLP (have to replace \n beforehand) -> use re.split w. multiple delimiters
def generate_context_samples(contexts: Collection[str], refs: Collection[str], 
                       meta: Dict[str, str], textpath: Path) -> DataFrame:
    samples = []
    for sentence in contexts:
        # return a list of all citations in a sentence
        hits = re.findall(CITATION_PATTERNS, sentence)
        for hit in hits:
            # remove the identifiers as we use them to split .refs file
            s = re.sub("GC|DBLP", '', hit)
            for ref in refs:
                if re.search(s[1:-1], ref):
                    # extract authors and preprocess
                    authors = re.findall(";(.*?)\`\`", ref)
                    authors = re.sub(r"\band\b", ',', authors)
                    authors = authors.split(',')
                    authors = [author.strip() for author in authors if len(author) > 3]
                    
                    # extract titles and preprocess
                    title = re.findall('\`\`(.*?)\'\'', ref)
                    
                    # generate sample in correct format
                    sample = {"context": re.sub(CITATION_PATTERNS, '', sentence),
                            "title_citing": meta["title"],
                            "authors_citing": ','.join(meta["authors"]),
                            "title_cited": title,
                            "authors_cited": authors}
                    samples.append(pd.DataFrame(sample, index=[0]))
    return samples


def clean_incomplete_data(path: PathOrStr) -> None:
    """
    Cleaning function for the arxiv CS dataset. Checks all .txt files in the target folder and looks
    for matching .ref and .meta files. If a file is missing, all others are deleted.  
    If any file of the 3 files (.txt, .meta, .refs) is empty, the triple is removed as well.     
    **Parameters**:   
    - *path* (PathOrStr): Path object or string to the dataset.      
    """
    path = Path(path)

    incomplete_paths = 0
    empty_files = 0
    no_files = len(list(path.glob("*.txt")))

    for textpath in path.glob("*.txt"):
        metapath = textpath.with_suffix(".meta")
        refpath = textpath.with_suffix(".refs")

        if ( not metapath.exists() ) or ( not refpath.exists() ):
            incomplete_paths += 1
            logging.debug(f"Found incomplete file: {textpath.stem}")
            textpath.unlink()
            try:
                metapath.unlink()
            except FileNotFoundError:
                pass
            try:
                refpath.unlink()
            except FileNotFoundError:
                pass
        else:
            with open(textpath, 'r') as f:
                text = f.read()
            with open(metapath, 'r') as f:
                meta = f.read()
            with open(refpath, 'r') as f:
                refs = f.read()

            if len(text) == 0 or len(meta) == 0 or len(refs) == 0:
                empty_files += 1
                logging.debug(f"Found empty file: {textpath.stem}")
                textpath.unlink()
                metapath.unlink()
                refpath.unlink()
    
    message = (f"Incomplete paths(not all files present): {incomplete_paths} out of {no_files}"
                f"\nAt least one empty file: {empty_files} out of {no_files}")
    logging.debug(message)


def prepare_data(path: PathOrStr) -> None:
    """ 
    Extracts citation contexts from each (.txt, .meta, .refs) tupel in the given location 
    and stores them in a DataFrame.  
    Each final sample has the form: [context, title_citing, authors_citing, title_cited, authors_cited].  
    The resulting DataFrame is saved as Python pickle object in the parent directory.  
    **Parameters**:   
    - *path* (PathOrStr): Path object or string to the dataset.
    """
    path = Path(path)
    save_dir = path.parent
    data = []

    for textpath in path.glob("*.txt"):
        metapath = textpath.with_suffix(".meta")
        refpath = textpath.with_suffix(".refs")

        with open(textpath, 'r') as f:
            text = f.read()
        with open(metapath, 'r') as f:
            meta = f.read()
        with open(refpath, 'r') as f:
            refs = f.read()
        
        # preprocess string data
        meta = json.loads(meta)
        text = process_text(text)
        refs = process_refs(refs)
        data.append(generate_context_samples(text, refs, meta, textpath))
    
    # prepare data for storage and save
    dataset = pd.concat(data, axis=0)
    dataset.reset_index(inplace=True)
    dataset.drop("index", axis=1, inplace=True)
    dataset.to_pickle(save_dir/"arxiv_data.pkl", compression=None)


def main():
    path_to_data = "/home/timo/DataSets/KD_arxiv_CS/arxiv-cs"
    # clean_incomplete_data(path_to_data)


if __name__ == '__main__':
    main()