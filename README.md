# Deep citation recommendation
PyTorch reimplementation of the neural citation network.
Authors original code can be found here:  
https://github.com/tebesu/NeuralCitationNetwork.
 
## Questions
* Embeddings: Rare words vs common words. How to choose vocab?
* Choose Max length for all info  
* How to embed contexts and titles?  
* Attention decoder help  
* We have no preselection algorithm. How to proceeed?    


## Stats  

1. Removed 8260 triplets of paper data due to empty/missing files.  
2. Removed 1 data sample throwing regex error.  
3. Removed 161670 context samples where information was missing/could not be parsed from files. 
4. Removed an additional 61 empty samples after preprocessing.  
* This leaves __502292 context - citation pairs__ with full information.
* __Context vocabulary__ size after processing: __120767__.  
* __Title vocabulary__ size after processing: __64128__.  
* Number of __citing authors__: __28582__.  
* Number of __cited authors__: __174241__. 