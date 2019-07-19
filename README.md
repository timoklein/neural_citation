# Neural Citation Network  
PyTorch reimplementation of the neural citation network.  

Author's source code:  
https://github.com/tebesu/NeuralCitationNetwork. 

Travis Ebesu, Yi Fang. Neural Citation Network for Context-Aware Citation Recommendation. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2017. PDF


## Getting started


## Data


## Stats  

1. Removed 8260 triplets of paper data due to empty/missing files.  
2. Removed 1 data sample throwing regex error.  
3. Removed 161670 context samples where information was missing/could not be parsed from files.   
* This leaves __502353 context - citation pairs__ with full information.
* __Context vocabulary__ size after processing: __72046__.  
* __Title vocabulary__ size after processing: __43208__.  
* Number of __citing authors__: __28200__.  
* Number of __cited authors__: __169236__.  


## Hyperparamters  

* Using author information = __True__
* Epochs = __10__  
* Batch size = __64__  
* GRU Hidden size = __128__  
* Recurrent layers # = __2__  
* Context filter sizes = __[4, 4, 5]__  
* Context # filters = __128__   
* Context embedding size = __128__  
* Maximum context length = __60__  
* Author filter sizes = __[1, 2]__  
* Authors # filters = __128__  
* Author embedding size = __128__  
* Maximum # of authors = __7__  
* Title embedding size = __128__  
* Maximum title length = __40__  
* Optimizer = __ADAM__  
* Learning rate = __0.001__  
* Dropout = __0.2__  
* Gradient clipping = __5.0__  

