import re
import pandas as pd
import logging
import json
from pandas import DataFrame
from pathlib import Path
from typing import Union, Collection, List, Dict
from collections import Counter
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from spacy.tokenizer import Tokenizer



logging.basicConfig(level=logging.INFO, style='$')

PathOrStr = Union[Path, str]
"""Custom type for Paths or pathlike objects."""

CITATION_PATTERNS = r"<DBLP:.*?>|<GC:.*?>"
"""Regex patterns for matching citations in document sentences."""


def process_text(text: str, delimiter: str = "\n============\n") -> List[str]:
    """
    Preprocessing function for preprocessing arxiv CS paper text.  

    ## Parameters:   

    - **text** *(str)*: .txt file string object containing the text of a paper.  
    - **delimiter** *(str = "\\n============\\n")*: token separating text sentences.  

    ## Output:  

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

    ## Parameters:   

    - **refs** *(str)*: reference file string.  
    - **delimiter_patterns** *(str = "GC|DBLP")*: regex patterns used to split the inidividual references.  

    ## Output:  

    - List citation contexts split at *delimiter*.
    """
    refs = re.sub("\n", '', refs)
    return re.split(delimiter_patterns, refs)



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
                try:
                    if re.search(s[1:-1], ref):
                        # find and preprocess authors
                        authors = re.findall(";(.*?)\`\`", ref)
                        authors = ''.join(authors)
                        authors = re.sub(r"\band\b", ',', authors)
                        authors = re.sub(r"-", '', authors)
                        authors = authors.strip(',  ')

                        # skip the sample if there is no author information
                        if len(authors) == 0:
                            continue
                        
                        # find and preprocess titles
                        title = re.findall('\`\`(.*?)\'\'', ref)
                        title = ''.join(title).strip(',')
                        
                        # generate sample in correct format
                        sample = {"title_citing": meta["title"],
                                "context": re.sub(CITATION_PATTERNS, '', sentence),
                                "authors_citing": ','.join(meta["authors"]),
                                "title_cited": title,
                                "authors_cited": authors}
                        samples.append(pd.DataFrame(sample, index=[0]))
                except:
                    logging.info('!'*30)
                    logging.info(f"Found erroneous ref at {textpath.stem}")
                    logging.info(ref)
    return samples


def clean_incomplete_data(path: PathOrStr) -> None:
    """
    Cleaning function for the arxiv CS dataset. Checks all .txt files in the target folder and looks
    for matching .ref and .meta files. If a file is missing, all others are deleted.  
    If any file of the 3 files (.txt, .meta, .refs) is empty, the triple is removed as well.  

    ## Parameters:   

    - **path** *(PathOrStr)*: Path object or string to the dataset.      
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
            logging.info(f"Found incomplete file: {textpath.stem}")
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
                logging.info(f"Found empty file: {textpath.stem}")
                textpath.unlink()
                metapath.unlink()
                refpath.unlink()
    
    message = (f"Incomplete paths(not all files present): {incomplete_paths} out of {no_files}"
                f"\nAt least one empty file: {empty_files} out of {no_files}")
    logging.info(message)


def prepare_data(path: PathOrStr) -> None:
    """ 
    Extracts citation contexts from each (.txt, .meta, .refs) tupel in the given location 
    and stores them in a DataFrame.  
    Each final sample has the form: [context, title_citing, authors_citing, title_cited, authors_cited].  
    The resulting DataFrame is saved as Python pickle object in the parent directory.  

    ## Parameters:   

    - **path** *(PathOrStr)*: Path object or string to the dataset.
    """
    path = Path(path)
    save_dir = path.parent
    if not save_dir.exists(): save_dir.mkdir()
    
    data = []

    no_total = len(list(path.glob("*.txt")))
    logging.info('-'*30)
    logging.info(f"Total number of files to process: {no_total}")
    logging.info('-'*30)

    for i, textpath in enumerate(path.glob("*.txt")):
        if i % 100 == 0: logging.info(f"Processing file {i} of {no_total}...")
        
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
        data.extend(generate_context_samples(text, refs, meta, textpath))


    dataset = pd.concat(data, axis=0)
    dataset.reset_index(inplace=True)
    dataset.drop("index", axis=1, inplace=True)
    dataset.to_pickle(save_dir/f"arxiv_data.pkl", compression=None)

def preprocess_dataset(path: PathOrStr) -> None:
    """
    Insert your description here.  
    
    ## Parameters:  
    
    - **param1** *(type)*:  
    
    ## Input:  
    
    - **Input 1** *(shapes)*:  
    
    ## Output:  
    
    - **Output 1** *(shapes)*:  
    """
    path = Path(path)
    data = pd.read_pickle(path)

    # prune empty fields
    data = data[(data["title_citing"] != "") & 
                (data["context"] != "") & 
                (data["authors_citing"] != "") & 
                (data["title_cited"] != "") & 
                (data["authors_cited"] != "")]
    
    # instantiate spacy model and preprocessers
    nlp = spacy.load("en_core_web_lg")
    tokenizer = Tokenizer(nlp.vocab)
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

    # preprocessing steps for contexts and cited titles
    for col in ["context", "title_cited"]:
        # lower case and strip
        data.loc[:, col] = data.loc[:, col].str.lower()
        data.loc[:, col] = data.loc[:, col].str.strip()
        # lemmatize, stopword removal, punctuation removal
        data.loc[:, col] = data.loc[:, col].apply(lambda x: [token.lemma_ for token in tokenizer(x) if not token.is_stop and not token.is_punct])
        # removing numbers and blanks
        data.loc[:, col] = data.loc[:, col].apply(lambda x: list(filter(lambda tok: tok.strip(), x)))
        data.loc[:, col] = data.loc[:, col].apply(lambda x: list(filter(lambda tok: tok.isalpha(), x)))

    # TODO: Print results upon complention (Vocab sizes)
    # TODO: Preprocessing for authors (do all steps except lemmatization -> check number of authors)

    data.to_pickle(path/f"processed_data.pkl", compression=None)



if __name__ == '__main__':
    # path_to_data = "/home/timo/DataSets/KD_arxiv_CS/arxiv-cs"
    # prepare_data(path_to_data)
    path_to_df = "/home/timo/DataSets/KD_arxiv_CS/arxiv_data.pkl"
    preprocess_dataset(path_to_df)