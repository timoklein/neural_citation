# Deep citation recommendation
PyTorch reimplementation of the neural citation network.
Authors original code can be found here:  
https://github.com/tebesu/NeuralCitationNetwork. 

## Stats  

1. Removed 8260 triplets of paper data due to empty/missing files.  
2. Removed 1 data sample throwing regex error.  
3. Removed 161670 context samples where information was missing/could not be parsed from files.   
* This leaves __502353 context - citation pairs__ with full information.
* __Context vocabulary__ size after processing: __72046__.  
* __Title vocabulary__ size after processing: __43208__.  
* Number of __citing authors__: __28200__.  
* Number of __cited authors__: __169236__.  

## Questions  

* Remove duplicates from corpus? [might affect recall!]  unique!  
* Check on if loss is calculated correctly [conditioned on true prior value]  True prior!  
* Presentation: 50/50 Split between powerpoint theory and jupyter part. 
     Jupyter part: data preprocessing specifics, attention, parameters, inference, documentation.
     Schwerpunkte: Motivation (warum das Modell, charakteristika), Approach, Attention, Recall,
        Losses (batchnorm vs nicht), keine Hyperparameter, Beispiel im Notebook 
        Datensatz (grobe Statistiken, Anzahl)

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
* Learning rate = __0.01__  
* Dropout = __0.2__  
* Gradient clipping = __5.0__  

## Experiment results  
* Batchnorm in TDNN: Seems to speed up convergence (todo: 2 end to end runs for comparison)  
* Custom initialization: Doesn't seem like big improvement, torch defaults are very good. 
* More TDNNs: TBD  
* weight decay: TBD  
