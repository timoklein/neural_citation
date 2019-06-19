# Deep citation recommendation
PyTorch reimplementation of the neural citation network.
Authors original code can be found here:  
https://github.com/tebesu/NeuralCitationNetwork.
 
## Questions
* Embeddings: Rare words vs common words. How to choose vocab?


## Stats  
* Remove 8260 triplets of paper data due to empty/missing files. 
* Remove 1 data sample throwing regex error.   
* Remove 161670 context samples where information was missing/could not be parsed from files.  
* This leaves 502353 context - citation pairs with full information.  