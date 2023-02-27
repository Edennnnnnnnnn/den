# den: Algorithm Library

##### - Author: `Eden Zhou`
##### - Current Version: `20230227`


## [ML] Machine Learning Algorithms
#### [KNN (K-nearest Neighbors Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN) [`SL:Classification`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch).py))
> ##### Requirements:
> - `PyTorch`, `pandas`, `numpy`;

(from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromSKLearn).py))
> - `from sklearn.neighbors import KNeighborsClassifier`;
> - CrossValidation applied;


#### [NB (Naive Bayes Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/NB) [`SL:Classification`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NB/NB(fromScratch).py))
> ##### Introduction:
> - Designed for NLP infomation(emotion/type) extraction purpose;
> ##### Requirements:
> - `math`, `copy`, `nltk`, `pandas`;

(from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NB/NB(fromSkLearn).py))
> - `from sklearn.naive_bayes import BernoulliNB`;
> - CrossValidation applied;


## [NLP] Natural Language Processing Algorithms
#### [NGrams (N-grams Language Model)](https://github.com/Edennnnnnnnnn/den/tree/main/NLP/NGrams) [`NLP` / `SL:Classification`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py))
> ##### Variants:
> - Method of [Unsmoothed](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L311)
> - Method of [Laplace Smoothing](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L358)
> - Method of [Interpolation Smoothing](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L406)
> ##### Requirements:
> - `nltk`, `numpy`;

#### [Brill Tagger](https://github.com/Edennnnnnnnnn/den/tree/main/NLP/BrillTagger) [`NLP:Tagger`] 
(from [NLTK](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/BrillTagger/BrillTagger(fromNLTK).py))
> - `from nltk.tag.brill_trainer import BrillTaggerTrainer`
> - `from nltk.tag.brill import Word, Pos`


## [SEARCH] Searching & Planning Algorithms
#### [Dijkstras (Dijkstra's Search)](https://github.com/Edennnnnnnnnn/den/tree/main/Search/Dijkstras) [`SEARCH:Space-based`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/Dijkstras/Dijkstras(fromScratch).py))
> ##### Requirements:
> - `heapq`, `numpy`;


#### [Bi-BS (Bidirectional Brute-force Search)](https://github.com/Edennnnnnnnnn/den/tree/main/Search/BiBS) [`SEARCH:Space-based`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/BiBS/BiBS(fromScratch).py))
> ##### Requirements:
> - `heapq`, `numpy`;


#### [A* (A* Search)](https://github.com/Edennnnnnnnnn/den/blob/main/algorithms/) [`SEARCH:Space-based`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/BiBS/BiBS(fromScratch).py))
> ##### Requirements:
> - `heapq`, `numpy`;


#### [Bi-A* (Bidirectional A* Search)](https://github.com/Edennnnnnnnnn/den/blob/main/algorithms/) [`SEARCH:Space-based`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/BiBS/BiBS(fromScratch).py))
> ##### Requirements:
> - `heapq`, `numpy`;


#### [MMBi-A* (Meet-in-the-middle Bidirectional A* Search)](https://github.com/Edennnnnnnnnn/den/tree/main/Search/MMBiAStar) [`SEARCH:Space-based`] 
(from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/MMBiAStar/MMBiAStar(fromScratch).py)
> ##### Requirements:
> - `heapq`, `numpy`;


#### [MCTS (Monte Carlo Tree Search)](https://github.com/Edennnnnnnnnn/den/blob/main/algorithms/) [`SEARCH:Space-based`] 


## [RL] Reinforcement Learning Algorithms
#### [MC]
#### [TD]
