# den: Algorithm Library

> #####  Author: `Eden Zhou`
> #####  Current Version: `20230616`
> #####  * ***For Sharing: Please star the repo before taking codes. Thanks for your support!!***


&nbsp;
## [ML] Machine Learning Algorithms

#### [HMMs (Hidden Markov Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/HMMs) [`SL:Classification`, `Generative`] 
> ##### (from [NLTK](https://github.com/Edennnnnnnnnn/den/blob/main/ML/HMMs/HMMs(fromNLTK).py))
> - Method: `from nltk.tag.hmm import HiddenMarkovModelTrainer, HiddenMarkovModelTagger`;

#### [k-means (k-means Clustering Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KMeans) [`UL:Clustering`, `Discriminative`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KMeans/KMeans.cs))
> - Intro: C# used, designed for game elements collecting based on the Unity Engr;
> - Requirements: `System`, `UnityEngine`;

#### [k-NN (k-nearest Neighbors Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN) [`SL:Classification`, `Discriminative`] 
> ##### (from [Scratch.1](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch.1).py))
> - Intro: Designed for pictures detection and classification;
> - Requirements: `pyTorch`, `pandas`, `torchvision`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN/KNN(Proj.1))
> ##### (from [Scratch.2A](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch.2)(A.Vectorization).py) [Scratch.2B](https://github.com/Edennnnnnnnnn/den/blob/main/ML/KNN/KNN(fromScratch.2)(B.Prediction).py))
> - Intro: Designed for document data classification and better retrieval, established based on SMART weighting scheme of `ltc`;
> - Requirements: `collections`, `csv`, `heapq`, `json`, `math`, `sys`, `nltk`;
> - Variants:
> - - Method of [Stopwords Pruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/KNN/KNN(fromScratch.2)(B.Prediction).py#L80)
> - - Method of [Punctuation Pruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/KNN/KNN(fromScratch.2)(B.Prediction).py#L93)
> - Relevant Project --> [Project.2](https://github.com/Edennnnnnnnnn/den/tree/main/ML/KNN/KNN(Proj.2))
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
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/LogisticRegression/LogisticRegression(fromScratch).py))
> - Requirements: `numpy`, `utils`(can be found in projects), `matplotlib.pyplot`;
> - Design: `Mini-batch Gradient Descent` applied for practical application (for projects), `MSE` is used for error analysis and all the learning process plotted;
> - - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/LogisticRegression/LogisticRegression(Proj.1))
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/LogisticRegression/LogisticRegression(fromSKLearn).py))
> - Method: `from sklearn.linear_model import LogisticRegression`;
> - Design: CrossValidation applied;

#### [NBs (Naive Bayes Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/NB) [`SL:Classification`, `Generative`]
> ##### (from [Scratch.1](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NBs/NBs(fromScratch.1).py))
> - Intro: Designed for NLP information (emotion/type) extraction purpose, with complete model analysis part (Confusion matrix based);
> - Requirements: `sys`, `csv`, `math`, `copy`, `nltk`, `pandas`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/NBs/NBs(Proj.1))
> ##### (from [Scratch.2A](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NBs/NBs(fromScratch.2)(A.Training).py) [Scratch.2B](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NBs/NBs(fromScratch.2)(B.Prediction).py))
> - Intro: Designed for document data classification and better retrieval;
> - Requirements: `json`, `sys`, `nltk`, `os`, `csv`, `nltk`;
> - Variants:
> - - Method of [Stopwords Pruning](https://github.com/Edennnnnnnnnn/den/blob/9e574eaf96374edf79024b200e323e446259413b/ML/NBs/NBs(fromScratch.2)(A.Training).py#L61)
> - - Method of [Punctuation Pruning](https://github.com/Edennnnnnnnnn/den/blob/9e574eaf96374edf79024b200e323e446259413b/ML/NBs/NBs(fromScratch.2)(A.Training).py#L74)
> - Relevant Project --> [Project.2](https://github.com/Edennnnnnnnnn/den/tree/main/ML/NBs/NBs(Proj.2))
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NBs/NBs(fromSkLearn).py))
> - Method: `from sklearn.naive_bayes import BernoulliNB`;
> - Design: Cross Validation applied;

#### [NNs (Basic Neural Networks)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/NNs)
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/NNs/NNs(fromScratch).py))
> - Intro: Established as a three-layer simple NN, `sigmoid` function is used as the activation function;
> - Requirements: `numpy`;

#### [Rocchio's Algorithm (Rocchio Classifier Model)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/Rocchio) [`SL:Classification`, `Discriminative`] 
> ##### (from [Scratch.1A](https://github.com/Edennnnnnnnnn/den/blob/main/ML/Rocchio/Rocchio(fromScratch)(A.Training).py) [Scratch.1B](https://github.com/Edennnnnnnnnn/den/blob/main/ML/Rocchio/Rocchio(fromScratch)(B.Prediction).py))
> - Intro: Designed for document data classification and better retrieval, established based on SMART weighting scheme of `ltc`;
> - Requirements: `os`, `collections`, `csv`, `heapq`, `json`, `math`, `sys`, `nltk`;
> - Variants:
> - - Method of [Stopwords Pruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/Rocchio/Rocchio(fromScratch)(A.Training).py#L64)
> - - Method of [Punctuation Pruning](https://github.com/Edennnnnnnnnn/den/blob/be789762f640d95e3baeca88fe8dd8884f43267f/ML/Rocchio/Rocchio(fromScratch)(A.Training).py#L77)
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/Rocchio/Rocchio(Proj.1))

#### [SVM (Support Vector Machine)](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SVM) [`SL:Classification`, `Discriminative`] 
> ##### (from [SKLearn](https://github.com/Edennnnnnnnnn/den/blob/main/ML/SVM/SVM(fromSKLearn).py))
> - Method: `from sklearn.svm import SVC`;
> - Design: Cross Validation applied;
> - Relevant Project --> [Project.1+](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SVM/SVM(Proj.1%2B))

#### [Softmax Regression](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SoftmaxRegression) [`SL:Classification`, `Discriminative`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/ML/SoftmaxRegression/SoftmaxRegression(fromScratch).py))
> - Intro: Designed for pictures detection and classification;
> - Requirements: `numpy`, `struct`, `matplotlib.pyplot`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/ML/SoftmaxRegression/SoftmaxRegression(Proj.1))


&nbsp;
## [NLP] Natural Language Processing Algorithms
#### [NGrams (N-grams Language Model)](https://github.com/Edennnnnnnnnn/den/tree/main/NLP/NGrams) [`NLP:Classic` / `SL:Classification`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py))
> - Variants:
> - - Method of [Unsmoothed](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L311);
> - - Method of [Laplace Smoothing](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L358);
> - - Method of [Interpolation Smoothing](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/NGrams/NGrams(fromScratch).py#L406);
> - Requirements: `nltk`, `numpy`;

#### [Brill Tagger](https://github.com/Edennnnnnnnnn/den/tree/main/NLP/BrillTagger) [`NLP:Tagger`] 
> ##### (from [NLTK](https://github.com/Edennnnnnnnnn/den/blob/main/NLP/BrillTagger/BrillTagger(fromNLTK).py))
> - Method: `from nltk.tag.brill_trainer import BrillTaggerTrainer`;
> - Method: `from nltk.tag.brill import Word, Pos`;
> - Method: `from nltk.tbl.template import Template`;


&nbsp;
## [AI] Searching, Planning & Optimization Algorithms

#### [Adam Optimization](https://github.com/Edennnnnnnnnn/den/tree/main/AI/AdamOptimization)
> ##### (from [Scratch(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/AI/AdamOptimization/AdamOptimization.ipynb))
> - Intro: Designed for RL function approximation; Adam optimizer is used in sec.2 for better hyper-param tuning;
> - Requirements: Jupyter-Notebook-Env. with `rl_glue.RLGlue`, `numpy`, `os`, `copy`, `matplotlib.pyplot`, `shutil`, ...;

#### [Alphabeta (Alphabeta Tree Search)](https://github.com/Edennnnnnnnnn/den/tree/main/AI/Alphabeta) [`Tree-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/Alphabeta/Alphabeta(fromScratch).py))
> - Requirements: `numpy`;
> - Variants:
> - - Version of [the Naive](https://github.com/Edennnnnnnnnn/den/blob/87ce79dd85303a225f7ecdb285a5ff39735069a5/AI/Alphabeta/Alphabeta(fromScratch).py#L14)
> - - Version of [the Depth-limited Heuristic](https://github.com/Edennnnnnnnnn/den/blob/87ce79dd85303a225f7ecdb285a5ff39735069a5/AI/Alphabeta/Alphabeta(fromScratch).py#L38)

#### [A* Search](https://github.com/Edennnnnnnnnn/den/tree/main/AI/AStar)  [`Space-based:Heuristic`] 
> ##### (from [Scratch.1](https://github.com/Edennnnnnnnnn/den/blob/main/AI/AStar/AStar(fromScratch.1).py))
> - Requirements: `heapq`, `numpy`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/AI/AStar/AStar(Proj.1))
> ##### (from [Scratch.2](https://github.com/Edennnnnnnnnn/den/blob/main/AI/AStar/AStar(fromScratch.2).py))
> - Intro: C# used, designed for game path finding based on the Unity Engr;
> - Requirements: `System`, `UnityEngine`;

#### [Bi-A* (Bidirectional A* Search)](https://github.com/Edennnnnnnnnn/den/tree/main/AI/BiAStar) [`Space-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/BiAStar/BiAStar(fromScratch).py))
> - Requirements: `heapq`, `numpy`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/AI/BiAStar/BiAStar(Proj.1))

#### [Bi-BS (Bidirectional Brute-force Search)](https://github.com/Edennnnnnnnnn/den/tree/main/AI/BiBS) [`Space-based:Uninfo`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/BiBS/BiBS(fromScratch).py))
> - Requirements: `heapq`, `numpy`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/AI/BiBS/BiBS(Proj.1))

#### [Dijkstra's Search](https://github.com/Edennnnnnnnnn/den/tree/main/AI/Dijkstras) [`Space-based:Uninfo`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/Dijkstras/Dijkstras(fromScratch).py))
> - Requirements: `heapq`, `numpy`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/AI/Dijkstras/Dijkstras(Proj.1))

#### [Minimax (Minimax Tree Search)](https://github.com/Edennnnnnnnnn/den/tree/main/AI/Minimax) [`Tree-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/Minimax/Minimax(fromScratch).py))
> - Requirements: `time`, `numpy`;
> - Variants:
> - - Version of [the Boolean](https://github.com/Edennnnnnnnnn/den/blob/d495a783323637a88f4ce1f65c023829c5490815/AI/Minimax/Minimax(fromScratch).py#L19)
> - - Version of [the Naive](https://github.com/Edennnnnnnnnn/den/blob/d495a783323637a88f4ce1f65c023829c5490815/AI/Minimax/Minimax(fromScratch).py#L56)

#### [MMBi-A* (Meet-in-the-middle Bidirectional A* Search)](https://github.com/Edennnnnnnnnn/den/tree/main/AI/MMBiAStar) [`Space-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/MMBiAStar/MMBiAStar(fromScratch).py))
> - Requirements: `heapq`, `numpy`;
> - Relevant Project --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/AI/MMBiAStar/MMBiAStar(Proj.1))

#### [Negamax (Negamax Tree Search)](https://github.com/Edennnnnnnnnn/den/tree/main/AI/Negamax) [`Tree-based:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/Negamax/Negamax(fromScratch).py))
> - Requirements: `time`, `numpy`;
> - Variants:
> - - Version of [the Boolean](https://github.com/Edennnnnnnnnn/den/blob/d495a783323637a88f4ce1f65c023829c5490815/AI/Negamax/Negamax(fromScratch).py#L23)
> - - Version of [the Naive](https://github.com/Edennnnnnnnnn/den/blob/d495a783323637a88f4ce1f65c023829c5490815/AI/Negamax/Negamax(fromScratch).py#L68)

#### [UCB (Upper Confidence Bound Algorithm)](https://github.com/Edennnnnnnnnn/den/tree/main/AI/UCB) [`Optimization:Heuristic`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/AI/UCB/UCB(fromScratch).py))
> - Requirements: `bernoulli`, `math`, `numpy`;



&nbsp;
## [IR] Information Retrieval Algorithms
#### [Inverted Indexing](https://github.com/Edennnnnnnnnn/den/tree/main/IR/InvertedIndex) 
> ##### (from [Scratch.A](https://github.com/Edennnnnnnnnn/den/blob/main/IR/InvertedIndex/InvertedIndex(fromScratch)(A.Indexing).py) [Scratch.B](https://github.com/Edennnnnnnnnn/den/blob/main/IR/InvertedIndex/InvertedIndex(fromScratch)(B.Querying).py))
> - Intro: Designed for JSON-based document keywords retrieval, precomputed and sorted for better index storage and querying;
> - Requirements: `re`, `sys`, `csv`, `json`, `os`;
> - Relevant Projects --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/IR/InvertedIndex/InvertedIndex(Proj.1))

#### [Regrex Date Indexing](https://github.com/Edennnnnnnnnn/den/tree/main/IR/RegrexDateIndex) 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/IR/RegrexDateIndex/RegrexDateIndex(fromScratch).py))
> - Intro: Designed for document date-keywords retrieval, precomputed and sorted for better index storage and querying;
> - Requirements: `re`, `os`, `sys`, `csv`, `nltk`;
> - Relevant Projects --> [Project.1](https://github.com/Edennnnnnnnnn/den/tree/main/IR/RegrexDateIndex/RegrexDateIndex(Proj.1))



&nbsp;
## [CD] Encoding, Decoding & Compresion Algorithms

#### [Huffman Coding](https://github.com/Edennnnnnnnnn/den/tree/main/CD/HuffmanCoding) [`Encoding/Decoding`] 
> ##### (from [Scratch](https://github.com/Edennnnnnnnnn/den/blob/main/CD/HuffmanCoding/HuffmanCoding(fromScratch).py))
> - Intro: Based on parsing-tree structure to enable fast encoding and decoding processes;



&nbsp;
## [RL] Reinforcement Learning Algorithms

#### [Dyna-Q & Dyna-Q+](https://github.com/Edennnnnnnnnn/den/tree/main/RL/Dyna-Q&Dyna-Q+) [`Model-based:Prediction`] 
> ##### (from [Scratch(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/Dyna-Q&Dyna-Q+/Dyna-Q&Dyna-Q+(fromScratch).ipynb))
> - Requirements: Jupyter-Notebook-Env. with `jdc`, `numpy`, `rl_glue.RLGlue`, `os`, `tqdm`, `matplotlib.pyplot` ...;

#### [Function Approximation & Sarsa](https://github.com/Edennnnnnnnnn/den/tree/main/RL/FunctionApproximation%26Sarsa) [`Model-based`] 
> ##### (from [Scratch(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/FunctionApproximation%26Sarsa/FunctionApproximation%26Sarsa(fromScratch).ipynb))
> - Requirements: Jupyter-Notebook-Env. with `rl_glue.RLGlue`, `numpy`, `os`, `itertools`, `matplotlib.pyplot`, `time` ...;

#### [Multi-armed Bandit Problem](https://github.com/Edennnnnnnnnn/den/tree/main/RL/MultiArmedBanditProblem)
> ##### (from [Scratch(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/MultiArmedBanditProblem/MultiArmedBanditProblem(fromScratch).md))
> - Requirements: Jupyter-Notebook-Env. with `rl_glue.RLGlue`, `numpy`, `time`, `matplotlib.pyplot`, ...;

#### [Policy Improvement, Value Improvement & Dynamic Programming (DP)](https://github.com/Edennnnnnnnnn/den/tree/main/RL/PolicyValueImprove%26DP) [`Model-based:Control`] 
> ##### (from [Scratch(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/PolicyValueImprove%26DP/PolicyValueImprove%26DP(fromScratch).ipynb))
> - Requirements: Jupyter-Notebook-Env. with `numpy`, ...;

#### [Q-Learning & Expected Sarsa](https://github.com/Edennnnnnnnnn/den/tree/main/RL/Q-Learning&ExpSarsa) [`Environment/Model-based:Control`] 
> ##### (from [Scratch(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/Q-Learning&ExpSarsa/Q-Learning&ExpSarsa(fromScratch).ipynb))
> - Requirements: Jupyter-Notebook-Env. with `scipy.stats`, `numpy`, `rl_glue.RLGlue`, `tqdm`, `matplotlib.pyplot` ...;

#### [Semi-gradient Temporal Difference (TD)](https://github.com/Edennnnnnnnnn/den/tree/main/RL/Semi-gradientTD) [`Environment-based:Prediction`] 
> ##### (from [Scratch.1(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/Semi-gradientTD/Semi-gradientTD(fromScratch.1).ipynb))
> - Requirements: Jupyter-Notebook-Env. with `rl_glue.RLGlue`, `tqdm`, ...;
> ##### (from [Scratch.2(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/Semi-gradientTD/Semi-gradientTD(fromScratch.2).ipynb))
> - Intro: With Neural Network Classifier applied;
> - Requirements: Jupyter-Notebook-Env. with `rl_glue.RLGlue`, `tqdm`, `os`, `matplotlib.pyplot`, `shutil`, ...;

#### [Temporal Difference (TD)](https://github.com/Edennnnnnnnnn/den/tree/main/RL/TD) [`Model-based:Prediction`] 
> ##### (from [Scratch(Notes)](https://github.com/Edennnnnnnnnn/den/blob/main/RL/TD/TD(fromScratch).ipynb))
> - Requirements: Jupyter-Notebook-Env. with `jdc`, `numpy`, `rl_glue.RLGlue`, `itertools`, `tqdm`, ...;
