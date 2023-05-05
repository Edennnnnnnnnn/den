"""
By Eden Zhou
Apr. 12, 2023
"""

import csv
import json
import math
import sys
import nltk

from nltk import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


class NaiveBayesModelPrediction:
    def __init__(self):
        """ Class Variables Initialized"""
        # Model I/O Settings:
        self.inputPath_testData: str = ""
        self.inputPath_model: str = ""

        # Model Learning Settings:
        self.categories: set = set()
        self.categoryNum: int = 0
        self.vocab: set = set()
        self.priors: dict = dict()
        self.likelihoods: dict = dict()
        self.prediction: dict = dict()
        self.hashTableTest: dict = {}
        self.analysis: dict = {}

    def setPaths(self, inputPath_testData: str, inputPath_model: str):
        """
            This is a setter function for storing the file paths in the model, these info will be used for inputs;
        :param
            - inputPath_testData, str, the file path for the input training data;
            - inputPath_model, str, the file path for the input model data;
        """
        self.inputPath_testData = inputPath_testData
        self.inputPath_model = inputPath_model

    def dataFilter(self, tokens: list, inStopWordsPruning=False, inPunctuationPruning=False) -> list:
        """
            This is a helper function for processing the data filtering.
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
            This is the data input method for the model; loads and preprocesses test data, in order to evaluate
        the performance of the model on the test data;
        :param
            - inStopWordsPruning, boolean, =True for removing all stopWords (Default: =False);
            - inPunctuationPruning, boolean, =True for killing all punctuation marks (Default: =False);
        """
        # Data are loaded based on JSON format and stored in the operator:
        with open(self.inputPath_testData, 'r', encoding='utf-8') as f:
            allData = json.loads(f.read())

        # Normalizing & pruning data input and rebuild the data structures:
        index = 0
        for sample in allData:
            category = sample.get("category")
            text = self.dataFilter([token.lower() for token in word_tokenize(sample.get("text"))],
                                   inStopWordsPruning=inStopWordsPruning,
                                   inPunctuationPruning=inPunctuationPruning)
            self.hashTableTest[str(index)] = [category, text]
            index += 1

        # Input model learned:
        with open(self.inputPath_model, "r") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for dataline in reader:
                if dataline['type'] == "prior":
                    self.categories.add(dataline['category'])
                    self.priors[dataline['category']] = math.log10(float(dataline['priorProb/term']))
                    self.likelihoods[dataline['category']] = {}
                elif dataline['type'] == "likelihood":
                    self.likelihoods[dataline['category']][dataline['priorProb/term']] = math.log10(float(dataline['~/likelihoodProb']))
                    self.vocab.add(dataline['priorProb/term'])
            self.categoryNum = len(self.categories)

    def modelPredict(self) -> None:
        """
            This function goes through the input test dataset to make predictions based on the previous knowledge leaned
        from the training dataset (the vocab, the likelihood and the prior); The algorithm of Naive Bayes classifier is
        used for this process, all prediction results are stored in a hashTable as a class/model variable; "log method"
        is continued to be used to avoid underflow issues and speed up the computation;
        """
        # Apply the formula of Naive Bayes classification to make predictions based on leaned knowledge:
        for sampleID in self.hashTableTest.keys():
            sampleTokens = self.hashTableTest[sampleID][1]
            counterForCategories = {}
            for category in self.categories:
                counterForCategories[category] = self.priors[category]
                for token in sampleTokens:
                    if token in self.vocab:
                        if token in self.likelihoods[category]:
                            counterForCategories[category] += self.likelihoods[category][token]
            self.prediction[sampleID] = max(counterForCategories, key=counterForCategories.get)

    def modelAnalyze(self):
        """
            This is the analysis and evaluation method for the model; This method builds up the confusion matrix
        (computing TP, FN, FP, TN, R and P values) based on relevant information for further method analysis; Three
        standards of precision analysis are applied (the pooled, the micro-averaged and the micro-averaged) for
        estimation;
        """
        # Deconstruct the data format and preprocess data for the confusion Matrix:
        truth_list = [self.hashTableTest.get(sampleID)[0] for sampleID in self.hashTableTest.keys()]
        prediction_list = [self.prediction.get(sampleID) for sampleID in self.prediction.keys()]
        classes = list(self.categories)

        # Initialize confusion matrix
        df_confusion = [[0] * len(classes) for _ in range(len(classes))]
        for truth, pred in zip(truth_list, prediction_list):
            i = classes.index(truth)
            j = classes.index(pred)
            df_confusion[i][j] += 1

        macro_p = 0
        macro_r = 0
        macro_f = 0
        pool = [float()] * 4
        analysis = {}

        for class_name, row in zip(classes, df_confusion):
            TP = row[classes.index(class_name)]
            FN = sum(row) - TP
            FP = sum([df_confusion[k][classes.index(class_name)] for k in range(len(classes))]) - TP
            TN = len(truth_list) - TP - FN - FP

            pool[0] += float(TP)
            pool[1] += float(FN)
            pool[2] += float(FP)
            pool[3] += float(TN)

            # Class precision & recall:
            precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            recall = TP / (TP + FN) if TP + FN > 0 else 0.0

            # Class F-values:
            macro_p += precision
            macro_r += recall
            class_F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            macro_f += class_F1
            analysis[class_name] = [TP, FP, FN, TN, precision, recall, class_F1]

        # Overall Micro/Macro-average Standards:
        micro_p = pool[0] / (pool[0] + pool[2]) if pool[0] + pool[2] > 0 else 0.0
        micro_r = pool[0] / (pool[0] + pool[1]) if pool[0] + pool[1] > 0 else 0.0
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0.0
        macro_f /= len(classes)
        analysis["overall"] = [micro_f, macro_f]

        self.analysis = analysis

    def modelOutput(self):
        """
            This is the evaluation output function for the model, it helps to print the analysis result to STDOUT flow
        within specific formats;
        """
        for category in self.analysis.keys():
            if category == "overall":
                continue
            categoryAnalysis = self.analysis.get(category)
            sys.stdout.write(f"\n* Category : {category} - TP={categoryAnalysis[0]}\tFP={categoryAnalysis[1]}\tFN={categoryAnalysis[2]}\tTN={categoryAnalysis[3]}\tPrecision={categoryAnalysis[4]}\trecall={categoryAnalysis[5]}\tF1={categoryAnalysis[6]}")
        sys.stdout.write(f"\n* Overall - Micro-averaged F1={self.analysis.get('overall')[0]}")
        sys.stdout.write(f"\n* Overall - Macro-averaged F1={self.analysis.get('overall')[1]}")


def main(argv, operator=NaiveBayesModelPrediction()):
    """
        The overall control function for the whole model project;
        Setting for Terminal Running with all kinds of error handling measures;
    :param:
        - argv, list, inputs from the command line, be used to locate data input and output paths;
        - operator, NaiveBayesModelPrediction, the model prediction process;
    """
    """ [Initializing] Handling all command parameters entered & Initializing the predictor """
    try:
        inputFile_model = argv[1]
        inputFile_testData = argv[2]
    except:
        # inputFile_model = "/Users/den/Desktop/CMPUT 361 - A3/w23-hw3-Edennnnnnnnnn/nbc_bbc_model.tsv"
        # inputFile_testData = "/Users/den/Desktop/CMPUT 361 - A3/w23-hw3-Edennnnnnnnnn/data/test.json"
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()

    """ [Basic Errors Handling] Initial checkers for input correctness """
    if inputFile_model == "" or inputFile_testData == "":
        sys.stderr.write("\n>> EmptyPathError: Input path(s) is/are invalid;")
        exit()
    if inputFile_model[-4:] != '.tsv':
        sys.stderr.write("\n>> FilePathError: The output file path is invalid; (Reminder: should end with '.tsv')")
        exit()
    if inputFile_testData[-5:] != '.json':
        sys.stderr.write("\n>> FilePathError: The input file path is invalid; (Reminder: should end with '.json')")
        exit()

    """ [Input] Acquire the indexing data input from the path provided and pre-processing the query"""
    try:
        operator.setPaths(inputPath_testData=inputFile_testData, inputPath_model=inputFile_model)
        operator.modelInput(inStopWordsPruning=True, inPunctuationPruning=True)
    except:
        sys.stderr.write("\n>> InputError: Failed to input given data to the model;")
        exit()

    """ [Processing] Parsing the query given and response to the querying task """
    try:
        operator.modelPredict()
    except:
        sys.stderr.write("\n>> ProcessingError: Failed to process model predicting based on given data;")
        exit()

    """ [Evaluate] Measure and report the model prediction quality"""
    try:
        operator.modelAnalyze()
    except:
        sys.stderr.write("\n>> AnalysisError: Failed to analyze the prediction data;")
        exit()

    """ [Output] Acquire the indexing data input from the path provided and pre-processing the query"""
    try:
        operator.modelOutput()
    except:
        sys.stderr.write("\n>> OutputError: Failed to output the prediction data;")
        exit()


if __name__ == "__main__":
    main(sys.argv)

    