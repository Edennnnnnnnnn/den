"""
By Eden Zhou
Apr. 12, 2023
"""

import collections
import csv
import heapq
import json
import math
import sys
import nltk

from nltk import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Reference: https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
#   - Used to solve _csv.Error：field larger than field limit；
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class RocchioModelPrediction:
    def __init__(self):
        """ Class Variables Initialized"""
        # Model I/O Settings:
        self.inputPath_testData: str = ""
        self.inputPath_model: str = ""

        # Model Parameter Settings:
        self.hashTableTest: dict = {}
        self.modelData: dict = {}
        self.modelIDF: dict = {}

        self.tokensTF: dict = {}
        self.ltcWeights: dict = {}
        self.prediction: dict = {}
        self.analysis: dict = {}

    def setPaths(self, inputPath_testData: str, inputPath_model: str):
        """
            This is a setter function for storing the file paths in the model, these info will be used for input and output;
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
            - tokens, list
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
            This is the data input method for the model which loads and preprocesses test data, in order to evaluate
        the performance of the model on the test data
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
                if dataline['type'] == "centroid":
                    #self.categories.add(dataline['category/term'])
                    categoryCentroid = {}
                    for centroidPoint in dataline['categoryCentroid/termIDF'].split('/ /'):
                        token, weight = centroidPoint.split(' ')
                        categoryCentroid[token] = float(weight)
                    self.modelData[dataline['category/term']] = categoryCentroid
                elif dataline['type'] == "idf":
                    self.modelIDF[dataline['category/term']] = float(dataline['categoryCentroid/termIDF'])

    def _getTFValues(self):
        """
            This is the method that calculates the term frequency (TF) values for tokens in a set of documents
        represented by a hash table; Counting the frequency of each token in the current document and stores the
        resulting dictionary.
        """
        # loop through each document in the hash table
        for docID in self.hashTableTest.keys():
            self.tokensTF[docID] = {}
            self.hashTableTest[docID][1] = dict(collections.Counter(self.hashTableTest[docID][1]))
            # loop through each token in the current document
            for token in self.hashTableTest[docID][1].keys():
                # calculate the TF value using the formula 1 + log(tf)
                self.tokensTF[docID][token] = 1 + math.log2(self.hashTableTest[docID][1][token])

    def _getltcWeights(self):
        """
            This code calculates the ltc weights for tokens in each document using their term frequency and inverse
        document frequency. It also normalizes the weights and stores them in a dictionary for each document.
        """
        # loop through each document in the tokensTF dictionary
        for docID in self.tokensTF.keys():
            # initialize a dictionary to store the ltc weights for tokens in the current document
            self.ltcWeights[docID] = {}

            # calculate the normalizer for the current document
            normalizer = 0
            for token in self.tokensTF[docID].keys():
                if self.modelIDF.get(token) is None:
                    self.ltcWeights[docID][token] = 0
                    continue

                # Calculate the weight of the token using the term frequency and IDF score
                c = self.tokensTF[docID][token]
                t = self.modelIDF[token]
                weight = c * t
                self.ltcWeights[docID][token] = weight
                normalizer += math.pow(weight, 2)
            normalizer = math.sqrt(normalizer)
            # loop through each token in the current document and calculate its ltc weight
            for token in self.ltcWeights[docID].keys():
                self.ltcWeights[docID][token] /= normalizer

    def _categoryFitting_measuring(self, testDocData: dict) -> dict:
        """
             This is the method that measures the distance between a test document and the model category centroids;
         Returns distanceRecorder, which maps each centroid to the distance between the test document and the centroid.
        :param
             - testDocData, dict, contains all test doc data for prediction;
        :return
            - distanceRecorder, dict, contains all distance data for each doc in prediction;
         """
        distanceRecorder = {}
        # Iterate through each token in the test document data
        for token in testDocData.keys():
            # Get the weight of the token in the test document
            testTokenWeight = testDocData[token]

            for modelCentroid in self.modelData.keys():
                # Get the weight of the token in the category centroid
                if self.modelData[modelCentroid].get(token) is not None:
                    modelTokenWeight = self.modelData[modelCentroid][token]
                    # Calculate the distance between the token weights in the test document and the category centroid
                    if distanceRecorder.get(modelCentroid) is None:
                        distanceRecorder[modelCentroid] = 0
                    distanceRecorder[modelCentroid] += abs(testTokenWeight - modelTokenWeight) ** 2

                # If the token is not present in the category centroid
                else:
                    # Calculate the distance between the token weight in the test document and 0
                    if distanceRecorder.get(modelCentroid) is None:
                        distanceRecorder[modelCentroid] = 0
                    distanceRecorder[modelCentroid] += abs(testTokenWeight) ** 2
        # Return the dictionary of distance values for each category centroid
        return distanceRecorder

    def _categoryFitting_ranking(self, testDocID: str, distanceRecorder: dict):
        """
            This function takes in a test document ID and a dictionary containing distances of the test
        document from each centroid;
        :param
            - testDocID: str, IDs of the test document;
            - distanceRecorder: dict, distances of the test document from each centroid;
        """
        distanceRanker = []
        for targetCentroid in distanceRecorder.keys():
            # Calculate the Euclidean distance between the test document and the current centroid.
            EuDist = math.sqrt(distanceRecorder[targetCentroid])
            # Push the distance and the corresponding centroid to the distanceRanker list.
            heapq.heappush(distanceRanker, (EuDist, targetCentroid))

        # Get the centroid with the smallest distance (i.e., closest to the test document)
        _, prediction = heapq.heappop(distanceRanker)

        # Assign the predicted centroid to the test document in the prediction dictionary.
        self.prediction[testDocID] = prediction

    def modelPredict(self) -> None:
        """
             This is the method that predicts the category of the test documents based on a training dataset.
         """
        self._getTFValues()
        self._getltcWeights()
        for docID_test in self.ltcWeights.keys():
            testDocData = self.ltcWeights[docID_test]
            self._categoryFitting_ranking(testDocID=docID_test,
                                          distanceRecorder=self._categoryFitting_measuring(testDocData=testDocData))

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
        classes = list(self.modelData.keys())

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


def main(argv, operator=RocchioModelPrediction()):
    """
        The overall control function for the whole model project;
        Setting for Terminal Running with all kinds of error handling measures;
    :param:
        - argv, list, inputs from the command line, be used to locate data input and output paths;
        - operator, RocchioModelPrediction, the model predict process
    """
    """ [Initializing] Handling all command parameters entered & Initializing the predictor """
    try:
        inputFile_model = argv[1]
        inputFile_testData = argv[2]
    except:
        # inputFile_model = "/Users/den/Desktop/CMPUT 361 - A3/w23-hw3-Edennnnnnnnnn/rocchio_bbc_model.tsv"
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
    