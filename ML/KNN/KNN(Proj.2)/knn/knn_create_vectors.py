import json
import math
import sys
import os
import csv
import collections
import nltk

from nltk import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


class KNNModelPreparation:
    def __init__(self):
        """ Class Variables Initialized"""
        # Model I/O Settings:
        self.inputPath_trainingData: str = ""
        self.outputPath_model: str = ""

        # Model Parameter Settings:
        self.hashTableTrain: dict = {}
        self.tokensIDF: dict = {}
        self.tokensTF: dict = {}
        self.ltcWeights: dict = {}

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
           - tokens, list, document filtered;
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
           - tokens, list, document filtered;
       """
        tokenizerPunc = RegexpTokenizer(r'\w+')
        tokens = tokenizerPunc.tokenize(' '.join(tokens))
        return tokens

    def modelInput(self, inStopWordsPruning=False, inPunctuationPruning=False):
        """
            This is the data input method for the model; loads and preprocesses training data in order to create a
        dictionary "hashTableTrain" that can be used for further analysis and modeling;
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
            self.hashTableTrain[str(index)] = [category, text]
            index += 1

    def _getIDFValues(self):
        """
            This is the method that calculates the inverse document frequency (IDF) values for tokens in a set of
        documents represented by a hash table; Initializes an empty dictionary called tokensIDF to store the IDF values
        for each token.
        """
        # get the number of documents in the hash table
        N = len(self.hashTableTrain.keys())

        # initialize a dictionary to store IDF values for tokens
        self.tokensIDF: dict = {}

        # loop through each document in the hash table
        for docID in self.hashTableTrain.keys():
            for token in list(set(self.hashTableTrain[docID][1])):
                # if the token is not in the IDF dictionary, set its value to 1
                if self.tokensIDF.get(token) is None:
                    self.tokensIDF[token] = 1
                self.tokensIDF[token] += 1

        # loop through each token in the IDF dictionary
        for token in self.tokensIDF.keys():
            self.tokensIDF[token] = math.log10(N / self.tokensIDF[token])

    def _getTFValues(self):
        """
            This is the method that calculates the term frequency (TF) values for tokens in a set of documents
        represented by a hash table; Counting the frequency of each token in the current document and stores the
        resulting dictionary.
        """
        # loop through each document in the hash table
        for docID in self.hashTableTrain.keys():
            self.tokensTF[docID] = {}
            # count the frequency of each token in the current document
            self.hashTableTrain[docID][1] = dict(collections.Counter(self.hashTableTrain[docID][1]))

            # loop through each token in the current document
            for token in self.hashTableTrain[docID][1].keys():
                self.tokensTF[docID][token] = 1 + math.log10(self.hashTableTrain[docID][1][token])

    def _getltcWeights(self):
        """
            Calculates the ltc weights for tokens in a set of documents. The code loops through each document in the
        collection and initializes a dictionary to store the ltc weights for tokens in that document
        """
        # loop through each document in the tokensTF dictionary
        for docID in self.tokensTF.keys():
            # initialize a dictionary to store the ltc weights for tokens in the current document
            self.ltcWeights[docID] = {}
            # calculate the normalizer for the current document
            normalizer = 0
            for token in self.tokensTF[docID].keys():
                c = self.tokensTF[docID][token]
                t = self.tokensIDF[token]
                weight = c * t
                self.ltcWeights[docID][token] = weight
                normalizer += math.pow(weight, 2)
            normalizer = math.sqrt(normalizer)

            # loop through each token in the current document and calculate its ltc weight
            for token in self.tokensTF[docID].keys():
                self.ltcWeights[docID][token] /= normalizer
        # delete the tokensTF dictionary to save memory
        del self.tokensTF

    def modelTrain(self):
        """
            Train a model using the (TF-IDF) algorithm. First calculates the term frequencies of each term in the corpus
        Then calculates the inverse document frequency of each term. Finally calculates the weights of each term in each
        document using the ltc weighting scheme
        """
        self._getTFValues()
        self._getIDFValues()
        self._getltcWeights()

    def modelOutput(self):
        """
            The output function generates an output file containing the ltc weights for each token in each document,
        as well as the idf values for each token in the collection. The output file is formatted as a tsv file with
        column names specified by the "header" list.
        """
        # Preparing the data format for output printing:
        printer = []
        for docID in self.hashTableTrain.keys():
            temp = []
            # Get the category and document tokens of the current document from the hash table
            category = self.hashTableTrain[docID][0]
            docTokens = self.hashTableTrain[docID][1]
            for token in docTokens.keys():
                tokenWeight = self.ltcWeights[docID][token]
                # Append the token and its ltc weight to the temporary list as a string
                temp.append(f"{str(token)} {str(tokenWeight)}")
            # Append the list containing the tokens and ltc weights for the current document to the main output list
            printer.append(["vector", category, '/ /'.join(temp)])

        for token in self.tokensIDF.keys():
            idf_t = self.tokensIDF[token]
            printer.append(["idf", token, idf_t])

        # Based on given paths and names, combining to get the full output path:
        header: list = ['type', 'category/term', 'docVector/termIDF']

        # Printing header and output data to specific .tsv files:
        with open(self.outputPath_model, 'w', newline='', encoding="utf8") as outputFile:
            writer = csv.writer(outputFile, delimiter="\t")
            writer.writerow(header)
            writer.writerows(printer)


def main(argv, operator=KNNModelPreparation()):
    """
        The overall control function for the whole model project;
        Setting for Terminal Running with all kinds of error handling measures;
    :param:
        - argv, list, inputs from the command line, be used to locate data input and output paths;
        - operator, KNNModelPreparation, the model preparation process
    """
    """ [Initializing] Handling all command parameters entered & Initializing the predictor """
    try:
        inputFile_trainingData = argv[1]
        outputFile_model = argv[2]
    except:
        # inputFile_trainingData = "/Users/den/Desktop/CMPUT 361 - A3/w23-hw3-Edennnnnnnnnn/data/train.json"
        # outputFile_model = "/Users/den/Desktop/CMPUT 361 - A3/w23-hw3-Edennnnnnnnnn/knn_bbc_vectors.tsv"
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()

    """ [Basic Errors Handling] Initial checkers for input correctness """
    # If the query is empty, raising an error:
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
            u = input("knn_bbc_vectors.tsv already exists at current folder, Do you want to overwrite it? (Y/N) ")
            while u.upper() != 'Y' and u.upper() != 'N':
                u = input("knn_bbc_vectors.tsv already exists at current folder, Do you want to overwrite it? (Y/N) ")
            if u.upper() == 'N':
                exit()
        operator.setPaths(inputPath_trainingData=inputFile_trainingData, outputPath_mode=outputFile_model)
        operator.modelInput(inStopWordsPruning=False, inPunctuationPruning=False)
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
