import sys
import csv
import math
from copy import deepcopy
from random import shuffle

import nltk
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

import pandas as pd


class NaiveBayesModel:
    def __init__(self, classes):
        """ Class Variables Initialized"""
        # Model Parameter Settings:
        self.classes = classes
        self.classNums = len(classes)

        # Model Learning Settings:
        self.vocab: set = set()
        self.priors: dict = dict()
        self.likelihoods: dict = dict()
        self.prediction: dict = dict()

        # Model Storage Settings:
        self.hashTableTrain = None
        self.hashTableTest = None

    def modelClear(self) -> None:
        """
            This function removes data storage parts of an existed model (can be trained), and keeps the knowledge
        leaned from the previous data; Those knowledge learned can be used for predictions on new input test data;
        :param
            - self;
        """
        self.hashTableTrain = None
        self.hashTableTest = None
        self.prediction: dict = dict()

    def modelInput(self, inputData: dict, dataType: str) -> None:
        """
            This is the data input method for the model; Filling self.hashTableTrain for training dataset or
        self.hashTableTest for tests;
        :param
            - self;
            - inputData, dict, the preprocessed data input;
            - dataType, str, corresponds to two possible types, for differentiating input data type;
        """
        # Differentiate input data type and fill into different storage parameters respectively:
        if dataType == "TRAIN":
            self.hashTableTrain = inputData
        elif dataType == "TEST":
            self.hashTableTest = inputData
        else:
            print("DataTypeError: Invalid input data type;")

    def modelTrain(self) -> None:
        """
            This is the training control method for the whole process; This function reads and processes training data
        input to compute main features of the Naive Bayes Model (the vocab, the likelihood and the prior); "log method"
        is used to avoid underflow issues and speed up the computation;
        :param
            - self;
        """
        # ** Data Structures:
        # * hashTableData:
        #   {id --> relation, ["token", "s"], (heads), (tails)}
        # * likelihoods:
        #   [{cls1: a}, {cls2: b}, {}, {}]
        # * priors:
        #   {cls1: 1, cls2: 2, cls3: 3, cls4: 4}

        # Initialize learning parameters:
        counterForClassTokens = dict()
        for cls in self.classes:
            caseDict = {}
            counterForClassTokens[cls] = caseDict
            self.likelihoods[cls] = caseDict

        # Read and classify class info:
        counterForClasses = [0] * self.classNums
        for sampleID in self.hashTableTrain.keys():
            sample = self.hashTableTrain.get(sampleID)
            sampleRelation = sample[0]
            sampleTokens = sample[1]

            assert sampleRelation in self.classes
            classIdx = self.classes.index(sampleRelation)
            counterForClasses[classIdx] += 1

            # Acquire Vocabulary corpus from the training dataset:
            for token in sampleTokens:
                self.vocab.add(token)
                if token in counterForClassTokens.get(sampleRelation).keys():
                    counterForClassTokens[sampleRelation][token] += 1
                else:
                    counterForClassTokens[sampleRelation][token] = 1

        # Go through the data set to compute prior values:
        totalSampleNums = sum(counterForClasses)
        i = 0
        while i < self.classNums:
            priorRaw = counterForClasses[i] / totalSampleNums
            priorLog = math.log(priorRaw)
            clsName = self.classes[i]
            self.priors[clsName] = priorLog
            i += 1

        # Go through the data set to compute likelihood values:
        for cls in self.classes:
            denominator = sum(
                counterForClassTokens.get(cls).get(token) for token in counterForClassTokens[cls].keys()) + len(
                self.vocab)
            for token in self.vocab:
                numerator = 1
                if token in counterForClassTokens.get(cls):
                    numerator += counterForClassTokens.get(cls)[token]
                self.likelihoods[cls][token] = math.log(numerator / denominator)

    def modelPredict(self) -> None:
        """
            This function goes through the input test dataset to make predictions based on the previous knowledge leaned
        from the training dataset (the vocab, the likelihood and the prior); The algorithm of Naive Bayes classifier is
        used for this process, all prediction results are stored in a hashTable as a class/model variable; "log method"
        is continued to be used to avoid underflow issues and speed up the computation;
        :param
            - self;
        """
        # Apply the formula of Naive Bayes classification to make predictions based on leaned knowledge:
        for sampleID in self.hashTableTest.keys():
            sample = self.hashTableTest.get(sampleID)
            sampleTokens = sample[1]
            counterForClasses = {}
            for cls in self.classes:
                counterForClasses[cls] = self.priors[cls]
                for token in sampleTokens:
                    if token in self.vocab:
                        if token in self.likelihoods[cls]:
                            counterForClasses[cls] += self.likelihoods[cls][token]
            self.prediction[sampleID] = max(counterForClasses, key=counterForClasses.get)

    def modelOutput(self) -> list:
        """
            The output function for the model; This function will return class predictions with relevant information
        of the input test dataset;
        :param
            - self;
        :return
            - [[ClassInGolden, ClassPredicted, SampleID], [...], ...], 2d list, model predictions with relevant infos;
        """
        # Collect necessary truth/prediction results of the test case with a unique ID, then return it as the output:
        return [[self.hashTableTest.get(sampleID)[0], self.prediction.get(sampleID), sampleID]
                for sampleID in self.hashTableTest.keys()]

    def modelAnalyze(self, reportNeeded=False, presentErrorDetails=False) -> float:
        """
            This is the analysis and evaluation method for the model; This method firstly evaluates the prediction
        accuracy (the gap between the goldenTruth and the modelPredictions), then builds up the confusion matrix based
        on relevant information for further method analysis; Three standards of precision analysis are applied (the
        pooled, the micro-averaged and the micro-averaged) for estimation; The error details will be printed to the
        terminal window if users selects the option of presentErrorDetails=True;
        :param
            - self
            - presentDetails, boolean, =True for printing details about errors in model prediction (Default: =False);
        :return
            - accuracy, float, accuracy computed of the current model;
        """
        # Deconstruct the data format for further accuracy analysis:
        truth_list = [self.hashTableTest.get(sampleID)[0] for sampleID in self.hashTableTest.keys()]
        prediction_list = [self.prediction.get(sampleID) for sampleID in self.prediction.keys()]

        print("\n\n\n\t\t\t\t** Evaluation Report **"
              "\n#################################################################")
        """ Part.1: Error Analysis """
        print(f"\nPart.1: Error Analysis\n"
              f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
              f"\nDetail of mis-classification could be found in the confusion matrix in Part.3")
        analyzer = tuple(zip(truth_list, prediction_list))
        correctNum, sumNum = 0, 0
        for truth, prediction in analyzer:
            if truth == prediction:
                if presentErrorDetails is True:
                    print("> Result = PASS")
                correctNum += 1
            else:
                if presentErrorDetails is True:
                    print("> Result = WRONG")
            sumNum += 1
            if presentErrorDetails is True:
                # When needed, print all the details of errors found:
                print(f"    * Truth = {truth}\n"
                      f"    * Prediction = {prediction}\n")

        """ Part.2: Accuracy Analysis """
        accuracy = correctNum / sumNum
        print(f"\n\n\nPart.2: Accuracy Analysis\n"
              f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
              f"* CorrectNum = {correctNum}\n"
              f"* SumNum = {sumNum}\n"
              f"* Accuracy = {accuracy}")

        """ Part.3: Confusion Matrix """
        y_actu = pd.Series(truth_list, name='Actual')
        y_pred = pd.Series(prediction_list, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred, margins=True)
        print("\n\n\nPart.3: Confusion Matrix")
        print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"[Confusion Matrix]\n-----------------------------------------------------------------")
        print(df_confusion)
        print(f"-----------------------------------------------------------------")

        """ Part.4: Precision & Recall """
        # Compute precision values (based on three standards):
        macro_p = 0
        macro_r = 0
        macro_f = 0
        pool = [float()] * 4  # [TP, FN, FP, TN]
        print("\n\n\nPart.4: Precision & Recall ")
        print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for class_name in self.classes:
            TP = df_confusion[class_name][class_name]
            FN = df_confusion['All'][class_name] - TP
            FP = df_confusion[class_name]['All'] - TP
            TN = df_confusion['All']['All'] - FN - FP - TP
            pool[0] += float(TP)
            pool[1] += float(FN)
            pool[2] += float(FP)
            pool[3] += float(TN)

            # class confusion matrix:
            print("\n\n* Class -->", class_name)
            print(f"-----------------------------------------------------------------")
            print("\t\t\t\tTrue " + class_name, "\t\t", "True not")
            print("System " + class_name, "\t", TP, "\t\t\t\t", FN)
            print("System not\t\t\t", FP, "\t\t\t\t", TN)
            print(f"-----------------------------------------------------------------")

            # precision & recall for each class:
            precision = TP / df_confusion[class_name]['All']
            recall = TP / df_confusion['All'][class_name]
            macro_p += precision
            macro_r += recall
            print(f"TP={TP}\t"
                  f"FN={FN}\t"
                  f"FP={FP}\t"
                  f"TN={TN}\t\t"
                  f"--> Sum={TP + FN + FP + TN}")
            print("precision: ", precision)
            print("recall: ", recall)

            # accuracy estimation for each class:
            class_accuracy = (TP + TN) / df_confusion['All']['All']
            print("accuracy: ", class_accuracy)
            class_F1 = 2 * precision * recall / (precision + recall)
            print("F1: ", class_F1)
            macro_f += class_F1


        """ Part.5: Micro/Macro-average Standards """
        # pool = [TP, FN, FP, TN]
        print("\n\n\nPart.5: Micro/Macro-average Standards")
        print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("[Overall Pooled Matrix]")
        print(f"-----------------------------------------------------------------")
        print("\t\t\tTrue yes", "\t", "True not")
        print("System yes", "\t", pool[0], "\t\t\t", pool[1])
        print("System not\t", pool[2], "\t\t\t", pool[3])
        print(f"-----------------------------------------------------------------")
        print(f"(Overall) TP={pool[0]}\t"
              f"FN={pool[1]}\t"
              f"FP={pool[2]}\t"
              f"TN={pool[3]}\t\t"
              f"--> Sum={pool[0] + pool[1] + pool[2] + pool[3]}")

        # micro & macro average
        micro_p = pool[0] / (pool[0] + pool[2])
        micro_r = pool[0] / (pool[0] + pool[1])
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)
        print("\n[Micro-average]")
        print("precision: ", micro_p, "\nrecall: ", micro_r, "\nF1: ", micro_f)

        print("\n[Macro-average]")
        print("precision: ", macro_p / self.classNums, "\nrecall: ", macro_r / self.classNums, "\nF1: ", macro_f / self.classNums)

        print("\n#################################################################")
        return micro_f


def fileReader(filename: str) -> dict:
    """
        This is a helper function for reading (input) and preprocessing the raw data read from the target input file
    path; The info read will be preprocessed and returned for coming model operations;
    :param
        - filename: str, the target input file path;
    :return
        - hashTableData: dict, data read and preprocessed;
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        dataInput = [dataline for dataline in reader]

        hashTableData = {}
        for dataline in dataInput[1:]:
            # data structure: {id --> type, "tokens", (heads), (tails)}
            hashTableData[dataline[0]] = [
                dataline[-3],
                word_tokenize(dataline[1].strip('\'')),
                word_tokenize(dataline[-2]),
                word_tokenize(dataline[-1])]
        return hashTableData


def dataFilter(dataset: dict, inParticularPruning=False, inStopWordsPruning=False, inPunctuationPruning=False) -> dict:
    """
       This is a helper function for processing the data filtering. dataset will be processed according to how filter
    combinations is applied to the dataset; _heuristicFilter_ParticularPruning will be applied if users selects the
    option of inParticularPruning=True which will extract the valuable information part heuristically based on given
    indexes; _heuristicFilter_StopWordsPruning will be applied if users selects the option of inStopWordsPruning=True
    which will remove all stopWords; and _heuristicFilter_PunctuationPruning will be applied if users selects the
    option of inPunctuationPruning=True which will kill all punctuation marks;
    :param
       - K: int, how many folds will be used for separation;
       - inParticularPruning: boolean, =True for extracting the valuable information part heuristically based on given
    indexes (Default: =False);
       - inStopWordsPruning: boolean, =True for removing all stopWords (Default: =False);
       - inPunctuationPruning: boolean, =True for killing all punctuation marks (Default: =False);
    :return
        - dataset, hashTable, same structure as the input dataset but filtered;
    """
    # * Data Structure:
    #   {id --> type, "tokens", (heads), (tails)}
    for sampleID in dataset.keys():
        tokens = dataset[sampleID][1]
        heads = dataset[sampleID][-2]
        tails = dataset[sampleID][-1]
        if inParticularPruning:
            tokens = Filters._heuristicFilter_ParticularPruning(tokens, heads, tails)
        if inStopWordsPruning:
            tokens = Filters._heuristicFilter_StopWordsPruning(tokens)
        if inPunctuationPruning:
            tokens = Filters._heuristicFilter_PunctuationPruning(tokens)
        dataset[sampleID][1] = tokens
    return dataset


class Filters:
    @staticmethod
    def _heuristicFilter_ParticularPruning(tokens, heads: list, tails: list) -> list:
        """
          This is a helper function which can extract the valuable information part heuristically based on given
       indexes;
       :param
          - tokens: list, document before filtering;
          - heads: list, indexes of valuable positions as heads in sentences;
          - tails: list, indexes of valuable positions as tails in sentences;
       :return
           - tokensPruned, list, document filtered;
       """
        # Initialize all heads/tails indexes (change to integer type):
        headsInt = [int(head) for head in heads]
        tailsInt = [int(tail) for tail in tails]

        if min(headsInt) > max(tailsInt):
            # Deal with the sequences from the tails to the heads:
            startIdx = max(tailsInt)
            endIdx = min(headsInt)
            isRegularSequence = False
        else:
            # Deal with the sequences from the heads to the tails:
            startIdx = max(headsInt)
            endIdx = min(tailsInt)
            isRegularSequence = True

        # Remove less valuable part and reconstruct the sentence:
        i = startIdx
        tokensPruned = []
        while i < endIdx:
            tokensPruned.append(tokens[i])
            i += 1

        if isRegularSequence:
            tokensPruned = heads + tokensPruned + tails
        else:
            tokensPruned = tails + tokensPruned + heads
        return tokensPruned

    @staticmethod
    def _heuristicFilter_StopWordsPruning(tokens: list) -> list:
        """
          This is a helper function which can remove all stopwords heuristically according to the nltk stopwords corpus;
       :param
          - tokens: list, document before filtering;
       :return
           - tokensFiltered, list, document filtered;
       """
        stopWords = set(stopwords.words('english'))
        tokensFiltered = [token for token in tokens if token not in stopWords]
        return tokensFiltered

    @staticmethod
    def _heuristicFilter_PunctuationPruning(tokens: list) -> list:
        """
          This is a helper function which can remove all punctuation marks heuristically based on regex;
       :param
          - tokens: list, document before filtering;
       :return
           - tokensFiltered, list, document filtered;
       """
        tokenizerPunc = RegexpTokenizer(r'\w+')
        tokensFiltered = tokenizerPunc.tokenize(' '.join(tokens))
        return tokensFiltered


class CrossValidationApproach:
    @staticmethod
    def kFoldsSeparator(K: int, dataset: dict, shuffleMode: bool = False) -> list:
        """
           This is a helper function for processing the separation part in the K-fold cross-validation approach; Based on
        the input k value, the function will cut the input training data into k parts and return them in a list; The
        shuffleModel will be applied if users selects the option of shuffleMode=True;
        :param
           - K: int, how many folds will be used for separation;
           - dataset: dict, the input pre-processed training dataset;
           - shuffleMode: boolean, =True for shuffling the dataset before dividing (Default: =False);
        :return
            - datasetReconstructed, 2d list, with k sub-lists inside as k tokens for taking the cross-validation process;
        """
        dataIDs = [sampleID for sampleID in dataset.keys()]
        if shuffleMode:
            shuffle(dataIDs)

        datasetReconstructed = []
        partsForIDs = [dataIDs[i::K] for i in range(K)]
        for partForIDs in partsForIDs:
            partReconstructed = {}
            for sampleID in partForIDs:
                partReconstructed[sampleID] = dataset.get(sampleID)
            datasetReconstructed.append(partReconstructed)
        return datasetReconstructed

    @staticmethod
    def modelOperator(K: int, folds: list, Model, classes, filterMode=False) -> list:
        """
           This is a helper function for processing the operation part in the K-fold cross-validation approach; Based on
        the input k value, the function will make a loop of k iterations and build up a user-defined model for each; The
        filterMode will be applied if users selects the option of shuffleMode=True;
        :param
           - K: int, how many folds will be used for separation;
           - folds: list, the datasetReconstructed from crossValidation_kFoldsSeparator;
           - Model, Object, a user-defined model object (class);
           - classes, list by default
           - filterMode: boolean, =True for applying filters in running cross-validation process (Default: =False);
        :return
            - [max(accuracies), bestPredictor], 2d list, accuracies and the corresponding bestPredictor; This setting is
        prepared for the ultraMode otherwise return only the bestPredictor will be good enough;
        """
        predictors = []
        accuracies = []
        i = 0
        while i < K:
            print(f"\n\n\n\n>> Current Training Iteration: {i}")
            foldsCopy = deepcopy(folds)
            testData = foldsCopy.pop(i)

            trainData = foldsCopy.pop(0)
            for j in range(len(foldsCopy)):
                trainData = {**trainData, **foldsCopy[j]}

            """ [ModelInit] Creating a specific model object for each iteration """
            predictor = Model(classes=classes)

            """ [ModelInput] Adding testing data set to the predictor """
            # data structure: {id --> type, "tokens", (heads), (tails)}
            if filterMode:
                predictor.modelInput(inputData=dataFilter(trainData,
                                                          inParticularPruning=inParticularPruning,
                                                          inStopWordsPruning=inStopWordsPruning,
                                                          inPunctuationPruning=inPunctuationPruning),
                                     dataType="TRAIN")
                predictor.modelInput(inputData=dataFilter(testData,
                                                          inParticularPruning=inParticularPruning,
                                                          inStopWordsPruning=inStopWordsPruning,
                                                          inPunctuationPruning=inPunctuationPruning),
                                     dataType="TEST")
            else:
                predictor.modelInput(inputData=trainData, dataType="TRAIN")
                predictor.modelInput(inputData=testData, dataType="TEST")

            """ [ModelProcessing] Model training and predicting """
            predictor.modelTrain()
            predictor.modelPredict()

            """ [ModelAnalyzing] Evaluating & Collecting the error in model predictions on the test dataset"""
            predictors.append(predictor)
            accuracies.append(predictor.modelAnalyze(reportNeeded=False, presentErrorDetails=False))
            i += 1
        bestPredictor: NaiveBayesModel = predictors[accuracies.index(max(accuracies))]
        bestPredictor.modelClear()
        return [max(accuracies), bestPredictor]


def fileWriter(filename: str, dataOutput: list) -> None:
    """
        This is a helper function for writing (output) the after-tagging prediction data to the target output file path;
    :param
        - filename: str, the target output file path;
        - dataOutput: list, the after-tagging prediction data;
    """
    # Printing all prediction results (stored in dataOutput) to the output file designed:
    header = ['original_label', 'output_label', 'row_id']
    with open(filename, 'w', newline='') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(header)
        writer.writerows(dataOutput)

    print("\n\n---------------------------------------------------------------------")
    print("Model Output Process: \n")
    print(f"--> You may find prediction results in a CSV file at [{filename}].\n")


def main(argv):
    """
        - The overall control function for the whole model project;
    :param
        - argv, list, Inputs from the command line, be used to locate data input and output paths;
    """

    """[Dashboard] User may adjust parameters in this part for the cross validation learning """
    # PARAMETER SETTINGS:
    k = 3  # --> k folds in cross-validation;
    t = 1  # --> t iterations of running cross-validation approaches, inUltraMode;
    Model = NaiveBayesModel
    classes = ["director", "performer", "publisher", "characters"]

    # LEARNING SETTINGS:
    inShuffleMode = False
    inUltraMode = False  # Warning: inShuffleMode=True is mandatory for inUltraMode=True;
    inFilterMode = False

    # FILTER SETTINGS:
    global inParticularPruning, inStopWordsPruning, inPunctuationPruning
    inParticularPruning = False  # Warning: inFilterMode=True is mandatory for pruning settings;
    inStopWordsPruning = False
    inPunctuationPruning = False

    nltk.download('stopwords')

    """ [Initializing] Handling all command parameters entered & Initializing the predictor """
    #  If users do not define the model type (optional fourth param), then applying the default model automatically;
    dataTrainFile = None
    dataTestFile = None
    dataOutputFile = None
    # Setting for General Running:
    paramIdx = 0
    while paramIdx < len(argv):
        if argv[paramIdx] == "--train":
            dataTrainFile = argv[paramIdx + 1]
        elif argv[paramIdx] == "--test":
            dataTestFile = argv[paramIdx + 1]
        elif argv[paramIdx] == "--output":
            dataOutputFile = argv[paramIdx + 1]
        paramIdx += 1
    # Parameter errors catcher:
    try:
        assert dataTrainFile is not None
        assert dataTestFile is not None
        assert dataOutputFile is not None
    except AssertionError:
        # print("ParameterError: Please checking command parameters;")
        # exit()
        # Setting for Manual Running:
        dataTrainFile = "/Users/den/Desktop/库 #20230317/CMPUT 497(461):501/CMPUT 497 - Assignments/CMPUT 497 - A5/f2021-asn5-Edennnnnnnnnn/data/train.csv"
        dataTestFile = "/Users/den/Desktop/库 #20230317/CMPUT 497(461):501/CMPUT 497 - Assignments/CMPUT 497 - A5/f2021-asn5-Edennnnnnnnnn/data/test.csv"
        dataOutputFile = "/Users/den/Desktop/库 #20230317/CMPUT 497(461):501/CMPUT 497 - Assignments/CMPUT 497 - A5/f2021-asn5-Edennnnnnnnnn/output/test.csv"

    """ [ModelProcessing] Apply k-Folds cross validation in training """
    if inUltraMode:
        predictors = [CrossValidationApproach.modelOperator(K=k,
                                                            folds=CrossValidationApproach.kFoldsSeparator(K=k,
                                                                                                          dataset=fileReader(dataTrainFile),
                                                                                                          shuffleMode=inShuffleMode),
                                                            Model=Model,
                                                            classes=classes,
                                                            filterMode=inFilterMode) for _ in range(t)]
        accuracies = [item[0] for item in predictors]
        predictors = [item[1] for item in predictors]
        predictor = predictors[accuracies.index(max(accuracies))]

    else:
        accuracy, predictor = CrossValidationApproach.modelOperator(K=k,
                                                                    folds=CrossValidationApproach.kFoldsSeparator(K=k,
                                                                                                                  dataset=fileReader(dataTrainFile),
                                                                                                                  shuffleMode=inShuffleMode),
                                                                    Model=Model,
                                                                    classes=classes,
                                                                    filterMode=inFilterMode)

    """ [ModelInput] Adding testing data set to the predictor """
    if inFilterMode:
        predictor.modelInput(inputData=dataFilter(fileReader(dataTestFile),
                                                  inParticularPruning=inParticularPruning,
                                                  inStopWordsPruning=inStopWordsPruning,
                                                  inPunctuationPruning=inPunctuationPruning),
                             dataType="TEST")
    else:
        predictor.modelInput(inputData=fileReader(dataTestFile), dataType="TEST")

    """ [ModelPredict] Training the models and make predictions for the test dataset """
    predictor.modelPredict()

    """ [ModelAnalyzing] Evaluate the error in model predictions and OOV words appeared in test dataset """
    print("\n\n\n>> FINAL TEST")
    predictor.modelAnalyze(reportNeeded=True, presentErrorDetails=False)

    fileWriter(filename=dataOutputFile, dataOutput=predictor.modelOutput())


if __name__ == "__main__":
    main(sys.argv)
