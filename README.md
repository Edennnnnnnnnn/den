# den: Algorithm Library

> #####  Author: `Eden Zhou`
> #####  Current Version: `20230505`

&nbsp;
## [ML] Machine Learning Algorithms

#### [HMMs (Hidden Markov Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/HMMs) [`SL:Classification`, `Generative`] 
> ##### (from [NLTK](https://github.com/Edennnnnnnnnn/den/blob/main/ML/HMMs/HMMs(fromNLTK).py))
> - Method: `from nltk.tag.hmm import HiddenMarkovModelTrainer, HiddenMarkovModelTagger`;

#### [KNN (k-nearest Neighbors Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN) [`SL:Classification`, `Discriminative`] 
> ##### (from [Scratch.1](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch.1).py))
> - Intro: Designed for pictures detection and classification;
> - Requirements: `pyTorch`, `pandas`, `torchvision`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN/KNN(Proj.1))
> ##### (from [Scratch.2A](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch.2)(A.Vectorization).py) [Scratch.2B](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch.2)(B.Prediction).py))
> - Intro: Designed for document data classification and better retrieval, established based on SMART weighting scheme of `ltc`;
> - Requirements: `collections`, `csv`, `heapq`, `json`, `math`, `sys`, `nltk`;
> - Variants:
> - - Method of [StopWordsPruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/KNN/KNN(fromScratch.2)(B.Prediction).py#L80)
> - - Method of [PunctuationPruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/KNN/KNN(fromScratch.2)(B.Prediction).py#L93)
> - Relevant Projects --> [Project.2](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN/KNN(Proj.2))
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromSKLearn).py))
> - Method: `from sklearn.neighbors import KNeighborsClassifier`;
> - Design: Cross Validation applied;

#### [Linear Regression](https://github.com/Edennnnnnnnnn/den/tree/main/ML/LinearRegression) [`SL:Classification`, `Discriminative`]
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/LinearRegression/LinearRegression(fromScratch).py))
> - Requirements: `numpy`, `utils`(can be found in projects), `matplotlib.pyplot`;
> - Design: `Mini-batch Gradient Descent` applied for practical application (for projects), `MSE` is used for error analysis and all the learning process plotted;
> - - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/LinearRegression/LinearRegression(Proj.1))
> - - Relevant Project --> [Project.2](https://github.com/Edennnnnnnnnn/den/tree/main/ML/LinearRegression/LinearRegression(Proj.2))

#### [Logistic Regression](https://github.com/Edennnnnnnnnn/den/tree/main/ML/LogisticRegression) [`SL:Classification`, `Discriminative`]
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/LogisticRegression/LogisticRegression(fromScratch).py)
> - Requirements: `numpy`, `utils`(can be found in projects), `matplotlib.pyplot`;
> - Design: `Mini-batch Gradient Descent` applied for practical application (for projects), `MSE` is used for error analysis and all the learning process plotted;
> - - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/LogisticRegression/LogisticRegression(Proj.1))
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/LogisticRegression/LogisticRegression(fromSKLearn).py))
> - Method: `from sklearn.linear_model import LogisticRegression`;
> - Design: CrossValidation applied;

#### [NBs (Naive Bayes Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/NB) [`SL:Classification`, `Generative`]
> ##### (from [Scratch.1](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NB/NB(fromScratch).py))
> - Intro: Designed for NLP information (emotion/type) extraction purpose, with complete model analysis part (Confusion matrix based);
> - Requirements: `sys`, `csv`, `math`, `copy`, `nltk`, `pandas`;
> - Relevant Projects --> [Project.1]()
> ##### (from [Scratch.2A](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/NB/NB(fromScratch.2)(A.Training).py) [Scratch.2B]())
> - Intro: Designed for document data classification and better retrieval;
> - Requirements: `json`, `sys`, `nltk`, `os`, `csv`, `nltk`;
> - Variants:
> - - Method of [StopWordsPruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/NB/NB(fromScratch.2)(A.Training).py#L61)
> - - Method of [PunctuationPruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/NB/NB(fromScratch.2)(A.Training).py#L74)
> - Relevant Projects --> [Project.2](https://github.com/Edennnnnnnnnn/den/tree/be789762f640d95e3baeca88fe8dd8884f43267f/ML/NB/NB(Proj.2))
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/NB/NB(fromSkLearn).py))
> - Method: `from sklearn.naive_bayes import BernoulliNB`;
> - Design: Cross Validation applied;

#### [Rocchio Model](https://github.com/Edennnnnnnnnn/den/tree/main/ML/Rocchio) [`SL:Classification`, `Discriminative`] 
> ##### (from [Scratch.A](https://github.com/Edennnnnnnnnn/den/blob/main/ML/Rocchio/Rocchio(fromScratch)(A.Training).py)) [Scratch.B](https://github.com/Edennnnnnnnnn/den/blob/main/ML/Rocchio/Rocchio(fromScratch)(B.Prediction).py)
> - Intro: Designed for document data classification and better retrieval, established based on SMART weighting scheme of `ltc`;
> - Requirements: `os`, `collections`, `csv`, `heapq`, `json`, `math`, `sys`, `nltk`;
> - Variants:
> - - Method of [StopWordsPruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/Rocchio/Rocchio(fromScratch)(A.Training).py#L64)
> - - Method of [PunctuationPruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/Rocchio/Rocchio(fromScratch)(A.Training).py#L77)
> - Relevant Projects --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/Rocchio/Rocchio(Proj.1))

#### [SVM (Support Vector Machine)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SVM) [`SL:Classification`, `Discriminative`] 
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/SVM/SVM(fromSKLearn).py))
> - Method: `from sklearn.svm import SVC`;
> - Design: Cross Validation applied;

#### [Softmax Regression](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SoftmaxRegression) [`SL:Classification`, `Discriminative`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/SoftmaxRegression/SoftmaxRegression(fromScratch).py))
> - Intro: Designed for pictures detection and classification;
> - Requirements: `numpy`, `struct`, `matplotlib.pyplot`;
> - > - Relevant Projects --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SoftmaxRegression/SoftmaxRegression(Proj.1))


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
## [AI] Searching, Planning & Simulation Algorithms
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
## [IR] Information Retrieval Algorithms
#### [Inverted Indexing]() 
> ##### (from [Scratch]())
> - Intro:;

&nbsp;
## [CD] Encoding, Decoding & Compresion Algorithms
#### [Huffman Coding](https://github.com/Edennnnnnnnnn/den/tree/main/CD/HuffmanCoding) [`Encoding/Decoding`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/CD/HuffmanCoding/HuffmanCoding(fromScratch).py))
> - Intro: Based on parsing-tree stucture to enable fast encoding and decoding processes;

&nbsp;
## [RL] Reinforcement Learning Algorithms

#### [MC]

#### [TD]
