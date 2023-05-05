import json
import sys
import os
import csv
import nltk

from nltk import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


class NaiveBayesModelTraining:
    def __init__(self):
        """ Class Variables Initialized"""
        # Model I/O Settings:
        self.inputPath_trainingData: str = ""
        self.outputPath_model: str = ""

        # Model Parameter Settings:
        self.categories: list = []
        self.categoryNum: int = 0
        self.vocab: set = set()
        self.priors: dict = dict()
        self.likelihoods: dict = dict()
        self.hashTableTrain: dict = {}

    def setPaths(self, inputPath_trainingData: str, outputPath_mode: str):
        """
           This is a setter function for storing the file paths in the model, these info will be used for input and output;
        :param
            - inputPath_trainingData, str, the file path for the input training data;
            - outputPath_mode, str, the output file path for the model data learned;
        """
        self.inputPath_trainingData = inputPath_trainingData
        self.outputPath_model = outputPath_mode

    def dataFilter(self, tokens: list, inStopWordsPruning=False, inPunctuationPruning=False) -> list:
        """
           This is a helper function for processing the data filtering;
        :param
            - tokens, list;
            - inStopWordsPruning, boolean, =True for removing all stopWords (Default: =False);
            - inPunctuationPruning, boolean, =True for killing all punctuation marks (Default: =False);
        :return
            - tokens, list, same structure as the input dataset but filtered;
        """
        if inStopWordsPruning is True:
            tokens = self._heuristicFilter_StopWordsPruning(tokens)
        if inPunctuationPruning is True:
            tokens = self._heuristicFilter_PunctuationPruning(tokens)
        return tokens

    @staticmethod
    def _heuristicFilter_StopWordsPruning(tokens: list) -> list:
        """
            This is a helper function which can remove all stopwords heuristically according to the nltk stopwords corpus;
        :param
            - tokens: list, document before filtering;
        :return
            - tokens, list, document tokens filtered;
        """
        stopWords = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stopWords]
        return tokens

    @staticmethod
    def _heuristicFilter_PunctuationPruning(tokens: list) -> list:
        """
            This is a helper function which can remove all punctuation marks heuristically based on regex;
        :param
            - tokens: list, document before filtering;
        :return
            - tokens, list, document tokens filtered;
        """
        tokenizerPunc = RegexpTokenizer(r'\w+')
        tokens = tokenizerPunc.tokenize(' '.join(tokens))
        return tokens

    def modelInput(self, inStopWordsPruning=False, inPunctuationPruning=False):
        """
            This is the data input method for the model that loads and preprocesses training data in order to create a
        dictionary that can be used for further analysis and modeling, as well as stores the categories of the data and
        their number for reference.
        :param
            - inStopWordsPruning, boolean, =True for removing all stopWords (Default: =False);
            - inPunctuationPruning, boolean, =True for killing all punctuation marks (Default: =False);
        """
        # Data are loaded based on JSON format and stored in the operator:
        with open(self.inputPath_trainingData, 'r', encoding='utf-8') as f:
            allData = json.loads(f.read())

        # Normalizing & pruning data input and rebuild the data structures:
        index = 0
        for sample in allData:
            category = sample.get("category")
            text = self.dataFilter([token.lower() for token in word_tokenize(sample.get("text"))],
                                   inStopWordsPruning=inStopWordsPruning,
                                   inPunctuationPruning=inPunctuationPruning)
            self.categories.append(category)
            self.hashTableTrain[str(index)] = [category, text]
            index += 1
        self.categories = list(set(self.categories))
        self.categoryNum = len(self.categories)

    def modelTrain(self):
        """
            This is the training control method for the whole process; Training the Naive Bayes classifier model.The
        method makes use of three data structures: hashTableData, likelihoods, and priors; Finally, the likelihoods and
        priors are stored in the likelihoods and priors dictionaries, respectively.
        """
        # Initialize learning parameters:
        counterForClassTokens = {}
        for cls in self.categories:
            caseDict = {}
            counterForClassTokens[cls] = caseDict
            self.likelihoods[cls] = caseDict

        # Read and classify class info:
        counterForClasses = [0] * self.categoryNum
        for sampleID in self.hashTableTrain.keys():
            sample = self.hashTableTrain.get(sampleID)
            sampleCategory = sample[0]
            sampleTokens = sample[1]

            assert sampleCategory in self.categories
            classIdx = self.categories.index(sampleCategory)
            counterForClasses[classIdx] += 1

            # Acquire Vocabulary corpus from the training dataset:
            for token in sampleTokens:
                self.vocab.add(token)
                if token in counterForClassTokens.get(sampleCategory).keys():
                    counterForClassTokens[sampleCategory][token] += 1
                else:
                    # Add-one Smoothing:
                    counterForClassTokens[sampleCategory][token] = 1

        # Go through the data set to compute prior values, with Laplace-smoothing:
        totalSampleNums = sum(counterForClasses)
        i = 0
        while i < self.categoryNum:
            prior = (counterForClasses[i] + 1) / (totalSampleNums + self.categoryNum)
            clsName = self.categories[i]
            self.priors[clsName] = prior
            i += 1

        # Go through the data set to compute likelihood values:
        for cls in self.categories:
            denominator = sum(counterForClassTokens.get(cls).get(token) for token in counterForClassTokens[cls].keys()) + len(self.vocab)
            for token in self.vocab:
                numerator = 1
                if token in counterForClassTokens.get(cls):
                    numerator += counterForClassTokens.get(cls)[token]
                self.likelihoods[cls][token] = numerator / denominator

    def modelOutput(self) -> None:
        """
            This function is responsible for generating the output of the model. Prints the prior probabilities and
        likelihood probabilities for each category and token to a specific TSV file. It also includes a header in the
        TSV file with the column names.
        """
        # Format preparing for model output:
        printer = []
        for category in self.priors.keys():
            printer.append(["prior", category, self.priors[category]])
        for category in self.likelihoods.keys():
            for token in self.likelihoods[category]:
                printer.append(["likelihood", category, token, self.likelihoods[category][token]])

        # Based on given paths and names, combining to get the full output path:
        header: list = ['type', 'category', 'priorProb/term', '~/likelihoodProb']
        # Printing header and output data to specific .tsv files:
        with open(self.outputPath_model, 'w', newline='', encoding="utf8") as outputFile:
            writer = csv.writer(outputFile, delimiter="\t")
            writer.writerow(header)
            writer.writerows(printer)


def main(argv, operator=NaiveBayesModelTraining()):
    """
        The overall control function for the whole model project.
        Setting for Terminal Running with all kinds of error handling measures;
    :param:
        - argv, list, inputs from the command line, be used to locate data input and output paths;
        - operator, NaiveBayesModelTraining, the model training process;
    """
    """ [Initializing] Handling all command parameters entered & Initializing the predictor """
    try:
        inputFile_trainingData = argv[1]
        outputFile_model = argv[2]
    except:
        # inputFile_trainingData = "/Users/den/Desktop/CMPUT 361 - A3/w23-hw3-Edennnnnnnnnn/data/train.json"
        # outputFile_model = "/Users/den/Desktop/CMPUT 361 - A3/w23-hw3-Edennnnnnnnnn/nbc_bbc_model.tsv"
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()

    """ [Basic Errors Handling] Initial checkers for input correctness """
    if inputFile_trainingData == "" or outputFile_model == "":
        sys.stderr.write("\n>> EmptyPathError: Input path(s) is/are invalid;")
        exit()
    if inputFile_trainingData[-5:] != '.json':
        sys.stderr.write("\n>> FilePathError: The input file path is invalid; (Reminder: should end with '.json')")
        exit()
    if outputFile_model[-4:] != '.tsv':
        sys.stderr.write("\n>> FilePathError: The output file path is invalid; (Reminder: should end with '.tsv')")
        exit()
        
    """ [Input] Ask for the permission to overwrite existing file, acquiring the indexing data input from the path provided and pre-processing the query"""
    try:
        if os.path.exists(outputFile_model):
            u = input("nbc_bbc_model.tsv already exists at current folder, Do you want to overwrite it? (Y/N) ")
            while u.upper() != 'Y' and u.upper() != 'N':
                u = input("nbc_bbc_model.tsv already exists at current folder, Do you want to overwrite it? (Y/N) ")
            if u.upper() == 'N':
                exit()
        operator.setPaths(inputPath_trainingData=inputFile_trainingData, outputPath_mode=outputFile_model)
        operator.modelInput(inStopWordsPruning=True, inPunctuationPruning=True)
    except:
        sys.stderr.write("\n>> InputError: Failed to input given data to the model;")
        exit()

    """ [Processing] Parsing the query given and response to the querying task """
    try:
        operator.modelTrain()
    except:
        sys.stderr.write("\n>> ProcessingError: Failed to process model training based on given data;")
        exit()

    """ [Output] Acquire the indexing data input from the path provided and pre-processing the query"""
    try:
        operator.modelOutput()
    except:
        sys.stderr.write("\n>> OutputError: Failed to output the model data;")
        exit()


if __name__ == "__main__":
    main(sys.argv)
