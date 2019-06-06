import re
import pandas as pd
import spacy
import logging
import json
from pandas import DataFrame
from pathlib import Path
from ast import literal_eval
from typing import Union, Collection, List, Dict


logging.basicConfig(level=logging.DEBUG, style='$')
PathOrStr = Union[Path, str]
CITATION_PATTERNS = r"<DBLP:.*?>|<GC:.*?>"

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

def process_text(text: str, delimiter: str = "\n============\n") -> List[str]:
    text = re.sub("<formula>", '', text)
    sentences = text.split(delimiter)
    contexts = []
    for sentence in sentences:
        if re.search(CITATION_PATTERNS, sentence):
            contexts.append(sentence)
    return contexts


def process_refs(refs: str, delimiter: str = "\n") -> List[str]:
    return refs.split(delimiter)


# TODO: FIX ref splitting at \n to GC and DBLP (have to replace \n beforehand) -> use re.split w. multiple delimiters
def generate_json_text(contexts: Collection[str], refs: Collection[str], 
                       meta: Dict[str, str], textpath: Path) -> DataFrame:
    samples = []
    for sentence in contexts:
        hits = re.findall(CITATION_PATTERNS, sentence)
        for hit in hits:
            for ref in refs:
                if re.search(hit[1:-1], ref):
                    author_idx = ref.find(';') + 1
                    data = ref[author_idx:]
                    try:
                        authors, title, *_ = data.split('.')
                        authors = re.sub(r"\band\b", ',', authors)
                        authors = authors.split(',')
                        authors = [author.strip() for author in authors if len(author) > 3]
                    except ValueError:
                        logging.info("Erroneous reference file found: " + textpath.stem)
                    try:
                        sample = {"context": re.sub(CITATION_PATTERNS, '', sentence),
                                "title_citing": meta["title"],
                                "authors_citing": ','.join(meta["authors"]),
                                "title_cited": title,
                                "authors_cited": ','.join(authors)}
                    except UnboundLocalError:
                        continue
                    samples.append(pd.DataFrame(sample, index=[0]))
    return samples


def clean_incomplete_data(path: PathOrStr) -> None:
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
            # textpath.unlink()
            # metapath.unlink()
            # refpath.unlink()
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
                # textpath.unlink()
                # metapath.unlink()
                # refpath.unlink()
    
    message = (f"Incomplete paths(not all files present): {incomplete_paths} out of {no_files}"
                f"\nAt least one empty file: {empty_files} out of {no_files}")
    logging.debug(message)




def prepare_data(path: PathOrStr) -> None:
    """
    Prepare the arxiv CS dataset and save in JSON format.
    INPUTS:
    * __path__(PathOrStr):         Path or string to files
    """
    path = Path(path)
    save_dir =Path("/home/timo/DataSets/KD_arxiv_CS")
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
        data.append(generate_json_text(text, refs, meta, textpath))
    
    # prepare data for storage and save
    dataset = pd.concat(data, axis=0)
    dataset.reset_index(inplace=True)
    dataset.drop("index", axis=1, inplace=True)
    dataset.to_pickle(save_dir/"arxiv_data", compression=None)



def main():
    path_to_data = "/home/timo/DataSets/KD_arxiv_CS/arxiv"
    clean_incomplete_data(path_to_data)

if __name__ == '__main__':
    main()