# Derivational_Task_MLP
We learn a multilayer perceptron model that generates vectors for the derived words, when given the vector for source word and the target affix. We also report the accuracy of the model after performing 5-fold cross validation. The glove vectors used in the task have been downloaded from http://nlp.stanford.edu/projects/glove/ (we have used the file with 6 billion tokens).

The files output by the program are :

1. AnsFastText.txt - fastText vectors of derived words in wordList.csv

2. AnsLzaridou.txt - Lazaridou vectors of the derived words in wordList.csv

3. AnsModel.txt - Vectors for derived words as provided by the model

The function 'derivedWordTask' returns 2 values : averaged cosine similarity between the corresponding words from output files 1 and 3, as well as 2 and 3.

The files used for the task are : 

1. Vector_lazaridou.txt : Word vectors for source and derived words as per the distributional space described in “Compositional-ly Derived Representations of Morphologically Complex Words in Distributional Semantics”

2. fastText_vectors.txt : Word vectors for source and derived words a per the fastText model

3. wordList.csv : CSV file containing the triplets (Source word, derived word and the affix)
