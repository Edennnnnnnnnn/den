# den: Algorithm Library

> #####  Author: `Eden Zhou`
> #####  Current Version: `20230227`

&nbsp;
## [ML] Machine Learning Algorithms
#### [KNN (K-nearest Neighbors Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN) [`SL:Classification`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch).py))
> - Requirements: `PyTorch`, `pandas`, `numpy`;
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromSKLearn).py))
> - Method: `from sklearn.neighbors import KNeighborsClassifier`;
> - Design: CrossValidation applied;


#### [NB (Naive Bayes Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/NB) [`SL:Classification`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NB/NB(fromScratch).py))
> - Intro: Designed for NLP infomation (emotion/type) extraction purpose;
> - Requirements: `math`, `copy`, `nltk`, `pandas`;
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NB/NB(fromSkLearn).py))
> - Method: `from sklearn.naive_bayes import BernoulliNB`;
> - Design: CrossValidation applied;


#### [HMM (Hidden Markov Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/HMMs) [`SL:Classification`] 
> ##### (from [NLTK](https://github.com/Edennnnnnnnnn/den/blob/main/ML/HMMs/HMMs(fromNLTK).py))
> - Method: `from nltk.tag.hmm import HiddenMarkovModelTrainer, HiddenMarkovModelTagger`;


#### [Logistic Regression](https://github.com/Edennnnnnnnnn/den/tree/main/ML/LogisticRegression) [`SL:Classification`] 
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/LogisticRegression/LogisticRegression(fromSKLearn).py))
> - Method: `from sklearn.linear_model import LogisticRegression`;
> - Design: CrossValidation applied;


#### [SVM (Support Vector Machine)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SVM) [`SL:Classification`] 
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/SVM/SVM(fromSKLearn).py))
> - Method: `from sklearn.svm import SVC`;
> - Design: CrossValidation applied;


&nbsp;
## [NLP] Natural Language Processing Algorithms
#### [NGrams (N-grams Language Model)](https://github.com/Edennnnnnnnnn/den/tree/main/NLP/NGrams) [`NLP` / `SL:Classification`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py))
> - Variants:
> - - Method of [Unsmoothed](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L311);
> - - Method of [Laplace Smoothing](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L358);
> - - Method of [Interpolation Smoothing](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L406);
> - Requirements: `nltk`, `numpy`;

#### [Brill (Brill Tagger)](https://github.com/Edennnnnnnnnn/den/tree/main/NLP/BrillTagger) [`NLP:Tagger`] 
> ##### (from [NLTK](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/BrillTagger/BrillTagger(fromNLTK).py))
> - Method: `from nltk.tag.brill_trainer import BrillTaggerTrainer`;
> - Method: `from nltk.tag.brill import Word, Pos`;
> - Method: `from nltk.tbl.template import Template`;

&nbsp;
## [SEARCH] Searching, Planning & Simulation Algorithms
#### [Dijkstras (Dijkstra's Search)](https://github.com/Edennnnnnnnnn/den/tree/main/Search/Dijkstras) [`Space-based:Uninfo`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/Dijkstras/Dijkstras(fromScratch).py))
> - Requirements: `heapq`, `numpy`;


#### [Bi-BS (Bidirectional Brute-force Search)](https://github.com/Edennnnnnnnnn/den/tree/main/Search/BiBS) [`Space-based:Uninfo`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/BiBS/BiBS(fromScratch).py))
> - Requirements: `heapq`, `numpy`;


#### [A* (A* Search)](https://github.com/Edennnnnnnnnn/den/blob/main/algorithms/)  [`Space-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/BiBS/BiBS(fromScratch).py))
> - Requirements: `heapq`, `numpy`;


#### [Bi-A* (Bidirectional A* Search)](https://github.com/Edennnnnnnnnn/den/blob/main/algorithms/) [`Space-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/BiBS/BiBS(fromScratch).py))
> - Requirements: `heapq`, `numpy`;


#### [MMBi-A* (Meet-in-the-middle Bidirectional A* Search)](https://github.com/Edennnnnnnnnn/den/tree/main/Search/MMBiAStar) [`Space-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/Search/MMBiAStar/MMBiAStar(fromScratch).py))
> - Requirements: `heapq`, `numpy`;


#### [MCTS (Monte Carlo Tree Search)](https://github.com/Edennnnnnnnnn/den/blob/main/algorithms/) [`Space-based:Heuristic`] 


&nbsp;
## [CD] Encoding, Decoding & Compresion Algorithms
#### [Huffman Coding](https://github.com/Edennnnnnnnnn/den/tree/main/CD/HuffmanCoding) [`Encoding/Decoding`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/CD/HuffmanCoding/HuffmanCoding(fromScratch).py))
> - Intro: Based on parsing-tree stucture to enable fast encoding and decoding processes;

&nbsp;
## [RL] Reinforcement Learning Algorithms

#### [MC]

#### [TD]
