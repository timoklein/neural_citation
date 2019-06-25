# Deep citation recommendation
PyTorch reimplementation of the neural citation network.
Authors original code can be found here:  
https://github.com/tebesu/NeuralCitationNetwork.
 
## Questions
* Embeddings: Rare words vs common words. How to choose vocab?
* Choose Max length for all info  
* How to embed contexts and titles?  
* Attention decoder help  


## Stats  
* Remove 8260 triplets of paper data due to empty/missing files. 
* Remove 1 data sample throwing regex error.   
* Remove 161670 context samples where information was missing/could not be parsed from files.  
* This leaves 502353 context - citation pairs with full information.  
* Context vocabulary size after processing: 120767.  
* Title vocabulary size after processing: 64128.  
* Number of citing authors: 28582.  
* Number of cited authors: 174241.  