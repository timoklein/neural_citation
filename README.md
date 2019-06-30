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
* Get Attention Decoder to work (Timo)  
* Convert Data into word vectors, build author embeddings (Timo)  
* Implement Bucketting (Timo)  
* Write theory (Joan)  
* Implement BM-25 (Joan)  

## Stats  

1. Removed 8260 triplets of paper data due to empty/missing files.  
2. Removed 1 data sample throwing regex error.  
3. Removed 161670 context samples where information was missing/could not be parsed from files. 
4. Removed an additional 61 empty samples after preprocessing.  
* This leaves __502292 context - citation pairs__ with full information.
* __Context vocabulary__ size after processing: __75027__.  
* __Title vocabulary__ size after processing: __44206__.  
* Number of __citing authors__: __28582__.  
* Number of __cited authors__: __174241__. 