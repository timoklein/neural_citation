import re
import pandas as pd
import logging
import json
import string
import spacy
import torch
from torch import Tensor
from pathlib import Path
from typing import Union, Collection, List, Dict
from collections import Counter
from functools import partial
from pandas import DataFrame
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from core import CITATION_PATTERNS, STOPWORDS, MAX_SEQ_LENGTH, MAX_AUTHORS, PathOrStr
import logging_setup



logger = logging.getLogger("neural_citation.data")


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
                    logger.info('!'*30)
                    logger.info(f"Found erroneous ref at {textpath.stem}")
                    logger.info(ref)
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
            logger.info(f"Found incomplete file: {textpath.stem}")
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
                logger.info(f"Found empty file: {textpath.stem}")
                textpath.unlink()
                metapath.unlink()
                refpath.unlink()
    
    message = (f"Incomplete paths(not all files present): {incomplete_paths} out of {no_files}"
                f"\nAt least one empty file: {empty_files} out of {no_files}")
    logger.info(message)


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
    logger.info('-'*30)
    logger.info(f"Total number of files to process: {no_total}")
    logger.info('-'*30)

    for i, textpath in enumerate(path.glob("*.txt")):
        if i % 100 == 0: logger.info(f"Processing file {i} of {no_total}...")
        
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


def title_context_preprocessing(text: str, tokenizer: Tokenizer) -> List[str]:
    """
    Applies the following preprocessing steps on a string:  

    1. Lowercase.  
    2. Remove all punctuation.  
    3. Tokenize.  
    4. Remove numbers.  
    5. Lemmatize.  
    6. Remove stopwords.  
    7. Remove blanks
  
    
    ## Parameters:  
    
    - **text** *(str)*: Text input to be processed.  
    - **tokenizer** *(spacy.tokenizer.Tokenizer)*: SpaCy tokenizer object used to split the string into tokens.   
    
    ## Output:  
    
    - **List of strings**:  List containing the preprocessed tokens.
    """
    text = text.lower().strip()
    text = re.sub("\d*?", '', text)
    text = re.sub("[" + re.escape(string.punctuation) + "]", " ", text)
    text = [token.lemma_ for token in tokenizer(text) if not token.like_num]
    text = [token for token in text if not token in STOPWORDS]
    text = [token for token in text if token.strip()]

    # return the sequence up to max length or totally if shorter
    try:
        return text[:MAX_SEQ_LENGTH]
    except IndexError:
        return text


def author_preprocessing(text: str) -> List[str]:
    """
    Applies the following preprocessing steps on a string:  

    
    1. Remove all numbers.  
    2. Lowercase.  
    3. Split at each comma.  
    4. Remove blanks.  
    5. Strip whitespace.  
    
    ## Parameters:  
    
    - **text** *(str)*: Text input to be processed.  
    
    ## Output:  
    
    - **List of strings**:  List containing the preprocessed author tokens. 
    """
    text = re.sub("\d*?", '', text)
    text = text.lower().split(',')
    text = [token.strip() for token in text if token.strip()]

    # return the sequence up to max length or totally if shorter
    try:
        return text[:MAX_AUTHORS]
    except IndexError:
        return text


def context_tensorizer(list_of_tokens: List[str], nlp: spacy.lang.en.English) -> Tensor:
    """
    Insert your description here.  
    
    ## Parameters:  
    
    - **param1** *(type)*:  
    
    ## Input:  
    
    - **Input 1** *(shapes)*:  
    
    ## Output:  
    
    - **Output 1** *(shapes)*:  
    """
    embedding_list = [nlp.vocab.get_vector(token) for token in list_of_tokens if token in nlp.vocab]
    t = torch.cat([torch.from_numpy(embd).unsqueeze(0) for embd in embedding_list], dim=0)
    return t


def preprocess_dataset(path_to_data: PathOrStr, 
                       context_title_cols: List[str] = ["context", "title_cited"],
                       author_cols: List[str] = ["authors_citing", "authors_cited"]) -> None:
    """
    Insert your description here.  
    
    ## Parameters:  
    
    - **param1** *(type)*:  
    
    ## Input:  
    
    - **Input 1** *(shapes)*:  
    
    ## Output:  
    
    - **Output 1** *(shapes)*:  
    """
    path_to_data = Path(path_to_data)
    data = pd.read_pickle(path_to_data)

    path_to_author_vecs = Path(path_to_author_vecs)

    # prune empty fields
    data = data[(data["title_citing"] != "") & 
                (data["context"] != "") & 
                (data["authors_citing"] != "") & 
                (data["title_cited"] != "") & 
                (data["authors_cited"] != "")]
    
    # instantiate spacy model and preprocessers
    nlp = spacy.load("en_core_web_lg")
    tokenizer = Tokenizer(nlp.vocab)

    preprocessor = partial(title_context_preprocessing, tokenizer=tokenizer)
    # preprocessing steps for contexts and cited titles
    for col in context_title_cols:
        data[col] = data[col].map(preprocessor)

    context_list = [item for sublist in data["context"].values.tolist() for item in sublist]
    title_list = [item for sublist in data["title_cited"].values.tolist() for item in sublist]
    context_counts = Counter(context_list)
    title_counts = Counter(title_list)
    msg = (f"Unique context tokens found: {len(context_counts)}"
           f"\nUnique title tokens found: {len(title_counts)}")
    logger.info(msg)


    for col in author_cols:
        data[col] = data[col].map(author_preprocessing)

    citing_list = [item for sublist in data["authors_citing"].values.tolist() for item in sublist]
    cited_list = [item for sublist in data["authors_cited"].values.tolist() for item in sublist]
    citing_counts = Counter(citing_list)
    cited_counts = Counter(cited_list)
    msg = (f"Unique citing authors tokens found: {len(citing_counts)}"
           f"\nUnique cited authors tokens found: {len(cited_counts)}")
    logger.info(msg)

    # augment dataframe with additional data
    data["context_len"] = data["context"].map(lambda x: len(x))
    data["title_len"] = data["title_cited"].map(lambda x: len(x))
    data["num_citing_aut"] = data["authors_citing"].map(lambda x: len(x))
    data["num_cited_aut"] = data["authors_cited"].map(lambda x: len(x))

    # reset the index to avoid indexing errors
    data.reset_index(drop=True, inplace=True)

    # drop incomplete data
    empty = pd.Index([])
    for idx in ["context_len", "title_len", "num_citing_aut", "num_cited_aut"]:
        empty = empty.union(data[data[idx] == 0].index)
    data.drop(empty, inplace=True)

    # reset index again
    data.reset_index(drop=True, inplace=True)

    data.to_pickle(path_to_data.parent/f"processed_data.pkl", compression=None)



if __name__ == '__main__':
    # path_to_data = "/home/timo/DataSets/KD_arxiv_CS/arxiv-cs"
    # prepare_data(path_to_data)
    path_to_df = "/home/timo/DataSets/KD_arxiv_CS/arxiv_data.pkl"
    path_to_aut_vecs = "/home/timo/DataSets/KD_arxiv_CS/embeddings/author_vecs.vec"
    preprocess_dataset(path_to_df, path_to_aut_vecs, tensorize_data=False)