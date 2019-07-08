# Deep citation recommendation
PyTorch reimplementation of the neural citation network.
Authors original code can be found here:  
https://github.com/tebesu/NeuralCitationNetwork.
 
## Questions
* Embeddings: Rare words vs common words. How to choose vocab?
* Choose Max length for all info  40 titles, 5 authors  
* How to embed contexts and titles?  First use Glove pretrained.  
* Attention decoder help  
* We have no preselection algorithm. How to proceeed?  BM-25 or whole test set.  
* Paperformat ACM template.  

## TODOs
* Implement training loops (Timo)
* Train model (Timo)
* Document code (Timo)
* Write theory (Joan)  
* Implement BM-25 (Joan)  

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
* Recurrent layers # = __1__  
* Context filter sizes = __[4, 4, 5]__  
* Context # filters = __128__   
* Context embedding size = __128__  
* Maximum context length = __100__  
* Author filter sizes = __[1, 2]__  
* Authors # filters = __128__  
* Author embedding size = __128__  
* Maximum # of authors = __7__  
* Title embedding size = __128__  
* Maximum title length = __40__  
* Optimizer = __ADAM__  
* Learning rate = __0.1__  
* Dropout = __0.8__  
* Gradient clipping = __5.0__  
