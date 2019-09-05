# Neural Citation Network  
PyTorch reimplementation of the neural citation network.  

Author's source code:  
https://github.com/tebesu/NeuralCitationNetwork. 

Travis Ebesu, Yi Fang. Neural Citation Network for Context-Aware Citation Recommendation. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2017. PDF

## Requirements
The neural_citation.yml file contains a list of packages used to implement this model. Of particular importance are:
* python==3.7.3
* pytorch==1.1.0
* torchtext==0.3.1
* tensorboard==1.14.0
* spacy==2.1.6
* gensim==3.8.0
* nltk==3.4.4
* pandas==0.25.0
* tqdm==4.32.2  

For training the model it is recommended to use the GPU version of Pytorch instead.

## Getting started  
Install dependencies and clone the repo. The training notebook contains a training  template.
The evaluation notebook shows an example of how model evaluations can be run.
Note that both training and evaluation functions are optimized for use in notebooks as they use
tqdm_notebook progress bars. When running these functions from a script this needs to be changed.  

## Project structure: 
    
    .
    ├── assets  # experiments in the EACL paper
    |   ├── various images # used for the NCN presentation
    │   ├── title_to_aut_cited # pickled dictionary: tokenized title -> cited authors
    │   └── title_tokenized_to_full # pickled dictionary: tokenized titles -> full cited paper titles
    │      
    ├── docs    # documentation for the ncn modules
    |    └── ncn # doc folder for the module
    │       └── xxx.html # doc html files
    |
    ├── ncn    # main folder containing the NCN implementation
    │   ├── core # contains core data types and constants
    │   ├── data # preprocessing pipeline for the dataset
    │   ├── evaluation # evaluator class for evaluation and inference
    │   ├── model # Neural Citation Network pytorch model
    │   └── training # low level pytorch training loops
    |
    ├── runs    # Example tensorboard training log 
    │   └── log folders # Folders containing a training run's logs
    |       └── train logs # tensorboard log files
    |
    ├── README.md     # This file
    ├── NCN_evaluation.ipynb # notebook for performing evaluation tasks
    ├── NCN_presentation.ipynb # presentation given about project, contains small inference demo
    ├── NCN_training.ipynb # training containing the high level training script
    └── neural_citation.yml # CPU conda environment




## Data  
The preprocessed dataset can be found here: https://drive.google.com/open?id=1qwBIXBsWp0ODrm91pVgBOwelJ5baGr-9.  
Statistics about the dataset can be found in the NCN_presentation.ipynb file.


## Weights
The model trained with the original settings by Ebesu and Fang can be found here: 
https://drive.google.com/open?id=1mT7kUb415wy0i1raXTPMcrXcWSgRC2pk.  

The model trained with our best configuration can be found here:

