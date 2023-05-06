import sys
import csv
import os
import math
from collections import Counter
from copy import deepcopy
from nltk import ngrams
from nltk.lm import Vocabulary


class NGramsModel:
    def __init__(self):
        """ Class Variables Initialized"""
        # Model Parameter Settings:
        self.prob: float = 0.0
        self.N: int = 8
        self.Ns = []
        self.models: list = ["--unsmoothed", "--laplace", "--interpolation"]
        self.model = self.models[0]

        # Model Storages Settings:
        self.hashTableTrain = {}
        self.hashTableDevelop = {}
        self.hashTableModel = {}
        self.hashTablePrediction = {}

    def setModel(self, modelInput: str = None) -> None:
        """
            This is the basic function for setting the model type. The function will get the target model type from the
        user input and apply it to the current model;
        :param:
            - self
            - modelInput, str, the target model type selected by users;
        """
        # The setter method for model type. Apply the user choice to the model setting;
        try:
            assert modelInput in self.models
        except AssertionError:
            print("InputError: invalid input model type, the default model 'unsmoothed' applied;")
        self.model = modelInput

    def setN(self, N: int) -> None:
        """
            This is the basic function for setting the N value. The function will acquire the target N value from the
        user input and apply it to the current model;
        :param:
            - self
            - N, int, the target N value for N-grams algorithm entered by users;
        """
        # The setter method for N value. Apply the user choice to the setting;
        self.N = N

    def setN_heuristic(self) -> None:
        """
            This is the heuristic function for setting the N value. The function will check the model selected by the
        user and automatically apply the best N value for model fitting to ensure the best prediction accuracy. Those
        smart N selections came from the prior knowledge in previous tests;
        :param:
            - self
        """
        # The automatic setter method for N value. Apply the heuristic choice (prior knowledge based) to the setting;
        if self.model == self.models[0]:
            self.N = 8
        elif self.model == self.models[1]:
            self.N = 3
        else:
            self.N = 3

    def modelInput(self, inputDirectoryPath: str, dataType: str = None) -> None:
        """
            This is the data input function for the whole program; Filling self.hashTableTrain for training or
        self.hashTableDevelop testing with the inputs data read from all valid files under the target directory path
        provided by users;
        :param:
            - self
            - inputDirectoryPath, str, the input directory path for the specific directory;
            - dataType, str, corresponds to the type of input, "T" --> Training and "D" --> Developing/Testing;
        """
        # Open specific data input files within the given directory path and record all their contexts:
        for inputFileName in os.listdir(inputDirectoryPath):
            with open(inputDirectoryPath + inputFileName, "r") as file:
                dataInFile = file.read().lower()
                if dataType == "T":
                    self.hashTableTrain[inputFileName] = dataInFile
                elif dataType == "D":
                    self.hashTableDevelop[inputFileName] = dataInFile
                else:
                    assert TypeError

    def modelInitialize(self, filename_train, N=None) -> None:
        """
            This function helps to set the training model and initialize the training data; All texts from teh training
        dataset will be processed and evaluated, self.hashTableModel will be filled with all the model data;
        :param:
            - self
            - filename_train, str, one of the training files' names, be used to track the data in self.hashTableTrain;
            - N, int/None(default), corresponds to the N value used for the N-gram model, which is not mandatory;
        """
        # Assertions and error handlers prepared to ensure the regular process;
        if N is None:
            N = self.N
        assert N >= 1

        # Initialize/pre-process the data input and deconstruct them to char-level expressions;
        trainChars = []
        for char in self.hashTableTrain.get(filename_train).replace("\n", " "):
            if char == (" " or "," or "."):
                continue
            trainChars.append(char)

        # Read all the training expressions to create basic features of the model and store them all;
        size = len(trainChars)
        vocab = Vocabulary(trainChars)
        nLevel_train = [''.join(token) for token in ngrams(trainChars, n=N)]
        nLevel_counter = Counter(nLevel_train)
        nm1Level_counter = nLevel_counter
        if N > 1:
            nm1Level_train = [''.join(token) for token in ngrams(trainChars, n=N - 1)]
            nm1Level_counter = Counter(nm1Level_train)
        self.hashTableModel[(filename_train, N)] = [vocab, size, nLevel_counter, nm1Level_counter]

    def modelTrain(self) -> None:
        """
            This is the training control method for the whole process; This function calls modelGeneralize function
        multiple times to set the training model and initialize the training data for a few possible cases; It will also
        check a few assertions to ensure the correctness of the model processing;
        :param:
            - self
        """
        # Assertions and error handlers prepared to ensure the regular process;
        assert self.N >= 1
        assert self.hashTableTrain is not None

        # Read each training datafile and apply self.modelInitialize method to build up general prediction models;
        for filename_train in self.hashTableTrain.keys():
            if self.model != self.models[2]:
                self.modelInitialize(filename_train)
            else:
                N = deepcopy(self.N)
                i = 0
                while i < 3:
                    if N == 0:
                        break
                    self.modelInitialize(filename_train, N)
                    N -= 1
                    i += 1

    def modelPredict_unsmoothed(self, filename_develop: str, testChars: list, testChars_size: int, N: int) -> None:
        """
            This function applies Unsmoothed N-gram Algorithm to process the model predictions; All the prediction
        results (contains the name of the model with the minimal Perplexity, and the value of the minimal Perplexity)
        will be recorded and loaded into self.hashTablePrediction for further operations;
        :param:
            - self
            - filename_develop, str, defines the current test/dev file for model testing/development;
            - testChars, list, contains all characters in the current test/dev file;
            - testChars_size, int, the length of testChars;
            - N, int, the N value;
        """
        # Find each model generalized in the hashtable, read their features and do the n-gram cutting based on N;
        MINPerp = math.inf
        MINModel = None
        for filename_model, N_model in self.hashTableModel.keys():
            assert N == N_model
            model = self.hashTableModel.get((filename_model, N))
            model_counter_n = model[2]
            model_counter_nm1 = model[3]

            testCombinations = ngrams(testChars, N)

            # AI/Count the num of test combinations in a model. Calculate prob and perp to evaluate each model;
            logProb = 0
            for testCombination in testCombinations:
                target_n = ''.join(testCombination)
                target_nm1 = ''.join(testCombination[:-1])
                C_n = model_counter_n.get(target_n)
                C_nm1 = model_counter_nm1.get(target_nm1)
                if C_n is None:
                    C_n = 0
                if C_nm1 is None:
                    C_nm1 = 0
                if N == 1:
                    C_nm1 = 1
                if (C_n == 0) or (C_nm1 == 0):
                    continue
                logProb += math.log2(C_n / C_nm1 * testChars_size)
            Perp = math.pow(2, -1 / testChars_size * logProb)

            # Find the model with the min perplexity, mark it as the best model and store it in the hashtable;
            if Perp < MINPerp:
                MINPerp = Perp
                MINModel = filename_model
        self.hashTablePrediction[(filename_develop, N)] = [filename_develop, MINModel, MINPerp, N]

    def modelPredict_laplace(self, filename_develop: str, testChars: list, testChars_size: int, N: int) -> None:
        """
            This function applies Laplace N-gram Algorithm to process the model predictions; All the prediction
        results (contains the name of the model with the minimal Perplexity, and the value of the minimal Perplexity)
        will be recorded and loaded into self.hashTablePrediction for further operations;
        :param:
            - self
            - filename_develop, str, defines the current test/dev file for model testing/development;
            - testChars, list, contains all characters in the current test/dev file;
            - testChars_size, int, the length of testChars;
            - N, int, the N value;
        """
        # Find each model generalized in the hashtable, read their features and do the n-gram cutting based on N;
        MINPerp = math.inf
        MINModel = None
        for filename_model, N_model in self.hashTableModel.keys():
            assert N == N_model
            model = self.hashTableModel.get((filename_model, N))
            model_vocab = model[0]
            model_counter_n = model[2]
            model_counter_nm1 = model[3]

            testCombinations = ngrams(testChars, N)

            # AI/Count the num of test combinations in a model. Calculate prob and perp to evaluate each model;
            logProb = 0
            for testCombination in testCombinations:
                target_n = ''.join(testCombination)
                target_nm1 = ''.join(testCombination[:-1])
                C_n = model_counter_n.get(target_n)
                C_nm1 = model_counter_nm1.get(target_nm1)
                if C_n is None:
                    C_n = 0
                C_n += 1
                if C_nm1 is None:
                    C_nm1 = 0
                C_nm1 += len(model_vocab)
                if N == 1:
                    C_nm1 = 1
                logProb += math.log2(C_n / C_nm1 * testChars_size)
            Perp = math.pow(2, -1 / testChars_size * logProb)

            # Find the model with the min perplexity, mark it as the best model and store it in the hashtable;
            if Perp < MINPerp:
                MINPerp = Perp
                MINModel = filename_model
        self.hashTablePrediction[(filename_develop, N)] = [filename_develop, MINModel, MINPerp, N]

    def modelPredict_interpolation(self, filename_develop, testChars, testChars_size, N) -> None:
        """
            This function applies Deleted Interpolation N-gram Algorithm to process the model predictions; Due to the
        property of Deleted Interpolation method, a heuristic method (self.getLambdas_heuristic(filename_develop,
        testChars)) about weights (Lambdas, shown in hashTableLambdas) is prepared to select weight parameters for each
        case smartly; All the prediction results (contains the name of the model with the minimal Perplexity, and the
        value of the minimal Perplexity) will be recorded and loaded into self.hashTablePrediction for further
        operations;
        :param:
            - self
            - filename_develop, str, defines the current test/dev file for model testing/development;
            - testChars, list, contains all characters in the current test/dev file;
            - testChars_size, int, the length of testChars;
            - N, int, the N value;
        """
        # Calling self.getLambdas_heuristic to get heuristic wight settings and models modified in new data structure;
        models, hashTableLambdas = self.getLambdas_heuristic(filename_develop, testChars)

        # Find each model generalized in the hashtable, read their features and do the n-gram cutting based on multi-Ns;
        MINPrep = math.inf
        MINModel = None
        for filename_model in models.keys():
            models_counter_n = []
            models_counter_nm1 = []
            models_Ns = models.get(filename_model)
            for model in models_Ns:
                if type(model) is int:
                    continue
                models_counter_n.append(model[2])
                models_counter_nm1.append(model[3])

            testCombinations = list(ngrams(testChars, N))

            # AI/Count the num of test combinations in Ns models. Calculate prob and perp to evaluate each group;
            logProb = 0
            for testCombination in testCombinations:
                currLambdas = hashTableLambdas.get((filename_develop, filename_model, testCombination))
                C_Ns = []  # [C_n, C_nm1, C_nm2]
                target_n = ''.join(testCombination)
                C_n = models_counter_n[0].get(target_n)
                C_Ns.append(C_n if C_n is not None else 0)
                if len(self.Ns) >= 2:
                    target_nm1 = ''.join(testCombination[:-1])
                    C_nm1 = models_counter_n[1].get(target_nm1)
                    C_Ns.append(C_nm1 if C_nm1 is not None else 0)
                    if len(self.Ns) >= 3:
                        target_nm2 = ''.join(testCombination[1])
                        C_nm2 = models_counter_n[2].get(target_nm2)
                        C_Ns.append(C_nm2 if C_nm2 is not None else 0)

                logProb_Ns = 0
                level = 0
                while level < len(self.Ns):
                    uppers = C_Ns[level]
                    if uppers == 0:
                        level += 1
                        continue
                    nextLevel = level + 1
                    limit = len(self.Ns)
                    if nextLevel < limit:
                        lowers = C_Ns[level + 1]
                    else:
                        lowers = 1
                    if lowers == 0:
                        continue
                    weight = currLambdas[level] if currLambdas is not None else 0
                    if weight != 0:
                        logProb_Ns += math.log2((uppers / lowers * testChars_size)) * weight
                    else:
                        logProb_Ns += math.log2((uppers / lowers * testChars_size))
                    level += 1
                logProb += logProb_Ns
            Perp = math.pow(2, -1 / testChars_size * logProb)

            # Find the model with the min perplexity, mark it as the best model and store it in the hashtable;
            if Perp < MINPrep:
                MINPrep = Perp
                MINModel = filename_model
        self.hashTablePrediction[(filename_develop, N)] = [filename_develop, MINModel, MINPrep, N]

    def getLambdas_heuristic(self, filename_develop, testChars) -> tuple:
        """
            This is a heuristic method which applies the Lambda selection Algorithm under the Deleted Interpolation 
        approach; This method is prepared to select weight parameters (Lambdas) for each combination cases in those 
        Interpolation N-grams predictions smartly;
        :param:
            - self
            - filename_develop, str, defines the current test/dev file for model testing/development;
            - testChars, list, contains all characters in the current test/dev file;
            - testChars_size, int, the length of testChars;
            - N, int, the N value;
        :returns:
            - (models, hashTableLambdas), tuple, contains models, which contains all models within the specific format
            (i.e., N-levels-based), and hashTableLambdas, the hashtable which stored all the values of Lambadas weights
            that can be used in the computation of the Probability/Perplexity in the method modelPredict_interpolation;
        """
        # Acquire all possible N values and store them in a class list variable reversely;
        N = deepcopy(self.N)
        self.Ns = []
        i = 0
        while i < 3:
            if N == 0:
                break
            self.Ns.append(N)
            N -= 1
            i += 1
        list(set(self.Ns)).sort(reverse=True)

        # Category models existed based on their N values; Put them into a new data structure for further operations;
        models = {}
        for filename_model, N_model in self.hashTableModel.keys():
            if filename_model not in models.keys():
                models[filename_model] = [0] * len(self.Ns)
            models[filename_model][self.Ns.index(N_model)] = self.hashTableModel.get((filename_model, N_model))

        # Read each filename of models in the hashtable; Acquire relevant Ns-grams combinations;
        hashTableLambdas = {}
        for filename_model in models.keys():
            models_Ns = models.get(filename_model)
            models_counter_Ns = [0] * len(self.Ns)
            index = 0
            for model_n in models_Ns:
                models_counter_Ns[index] = model_n[2]
                index += 1
            models_size = models_Ns[self.Ns.index(self.N)][1]
            testCombinations = ngrams(testChars, self.N)

            # AI/Count the num of types of combinations; Then compute weights in Deleted Interpolation approach;
            Lambdas = [0] * len(self.Ns)
            for testCombination in testCombinations:
                target_case1_nu = ''.join(testCombination)
                target_case1_de = ''.join(testCombination[:-1])
                target_case2_nu = ''.join(testCombination[1:])
                target_case2_de = ''.join(testCombination[1])
                target_case3_nu = ''.join(testCombination[-1])

                C_case1_nu_catcher = models_counter_Ns[self.Ns.index(self.N)].get(target_case1_nu)
                C_case1_nu = C_case1_nu_catcher if C_case1_nu_catcher is not None else 0
                if C_case1_nu == 0:
                    continue
                C_case1_de_catcher = models_counter_Ns[self.Ns.index(self.N - 1)].get(target_case1_de)
                C_case1_de = C_case1_de_catcher if C_case1_de_catcher is not None else 0
                case_1 = (C_case1_nu - 1) / (C_case1_de - 1) if (C_case1_de - 1) != 0 else 0

                C_case2_nu_catcher = models_counter_Ns[self.Ns.index(self.N - 1)].get(target_case2_nu)
                C_case2_nu = C_case2_nu_catcher if C_case2_nu_catcher is not None else 0
                C_case2_de_catcher = models_counter_Ns[self.Ns.index(self.N - 2)].get(target_case2_de)
                C_case2_de = C_case2_de_catcher if C_case2_de_catcher is not None else 0
                case_2 = (C_case2_nu - 1) / (C_case2_de - 1) if (C_case2_de - 1) != 0 else 0

                C_case3_nu_catcher = models_counter_Ns[self.Ns.index(self.N - 1)].get(target_case3_nu)
                C_case3_nu = C_case3_nu_catcher if C_case3_nu_catcher is not None else 0
                C_case3_de = models_size
                case_3 = (C_case3_nu - 1) / (C_case3_de - 1) if (C_case3_de - 1) != 0 else 0

                cases = [case_1, case_2, case_3]
                maxOne = max(cases)
                level = cases.index(maxOne)
                Lambdas[level] += C_case1_nu

                # Normalize the weights; Store them for each combination in each test case under each model with Ns;
                total = sum(Lambdas)
                if total == 0:
                    continue
                Lambdas[0] /= total
                Lambdas[1] /= total
                Lambdas[2] /= total
                hashTableLambdas[(filename_develop, filename_model, testCombination)] = Lambdas
        return models, hashTableLambdas

    def modelPredict(self) -> None:
        """
            This is the prediction control method for the whole process; This function calls different functions with
        diverse variant methods of N-gram Algorithm based on the user settings; It will also check a few assertions to
        ensure the correctness of the model processing;
        :param:
            - self
        """
        # Assertions and error handlers prepared to ensure the regular process;
        assert self.N >= 1
        assert self.hashTableModel is not None
        assert self.hashTableDevelop is not None

        # Initialize/pre-process the data input and deconstruct them to char-level expressions;
        for filename_develop in self.hashTableDevelop.keys():
            testChars = []
            for char in self.hashTableDevelop.get(filename_develop).replace("\n", " "):
                if char == (" " or "," or "."):
                    continue
                testChars.append(char)
            testChars_size = len(testChars)

            # Select/Apply the type selected by the user to do predictions;
            if self.model == self.models[0]:
                self.modelPredict_unsmoothed(filename_develop=filename_develop,
                                             testChars=testChars,
                                             testChars_size=testChars_size,
                                             N=self.N)
            elif self.model == self.models[1]:
                self.modelPredict_laplace(filename_develop=filename_develop,
                                          testChars=testChars,
                                          testChars_size=testChars_size,
                                          N=self.N)
            else:
                self.modelPredict_interpolation(filename_develop=filename_develop,
                                                testChars=testChars,
                                                testChars_size=testChars_size,
                                                N=self.N)

    def modelAnalyze(self, printFeedbacks: bool = False) -> None:
        """
            This is the analysis and evaluation method for the whole process; This function combines all outputs from
        any prediction function, and evaluates the model prediction accuracy based on some prior knowledge provided as
        the data labels (i.e., the filenames of the training/testing data); The analysis report will be printed to the
        terminal window with all the relevant information about the current model; Users may select the option of
        printFeedbacks=True to own mode details in processing; This method is not mandatory for the project;
        :param:
            - self
            - printFeedbacks, boolean, =True for printing the precessing details in model prediction (Default: =False);
        """
        # Print Feedbacks (details in processing), Bias Report (errors in prediction) and Analysis Report (accuracy);
        predictions = self.hashTablePrediction.values()
        successPredictionNum = 0
        errors = []
        totalNum = 0
        if printFeedbacks:
            print("\n\n---------------------------------------------------------------------")
            print("Model Feedbacks: \n")
        for prediction in predictions:
            if printFeedbacks:
                print(prediction)
            devlopFilename = prediction[0].split(".")[0]
            modelFilename = prediction[1].split(".")[0]
            if devlopFilename == modelFilename:
                successPredictionNum += 1
            else:
                errors.append((devlopFilename, modelFilename))
            totalNum += 1
        print("\n\n---------------------------------------------------------------------")
        print("Model Bias Report: \n")
        for item in errors:
            print(item)
        print("\n\n---------------------------------------------------------------------")
        print("Model Analysis Report: \n")
        print(f"* modelInfo = {self.model} : {self.N}")
        print(f"* successPredictionNum = {successPredictionNum}")
        print(f"* totalNum = {totalNum}")
        print(f"* accuracy = {successPredictionNum / totalNum}")

    def modelOutput(self, pathOutput) -> None:
        """
            The output function for the whole program; This function will follow the regular printing format to output
        the model prediction results to the target CSV file selected by the user;
        :param:
            - dataOutput,
            - pathOutput, str, defines the target output path;
        """
        # Selecting the type of model used to fit the output filename (path):
        # projectPath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".") + '/'
        
        """
        if pathOutput is None:
            if self.model == self.models[0]:
                outputFile = "/output/results_dev_unsmoothed.csv"
                # outputFile = projectPath + pathOutput
            elif self.model == self.models[1]:
                outputFile = "/output/results_dev_laplace.csv"
                # outputFile = projectPath + pathOutput
            else:
                outputFile = "/output/results_dev_interpolation.csv"
                # outputFile = projectPath + pathOutput
        else:
            # outputFile = projectPath + pathOutput
            outputFile = pathOutput
        print(pathOutput)
        """
        outputFile = pathOutput

        # Printing all prediction results (stored in dataOutput) to the output file designed:
        header = ['Testing_file', 'Training_file', 'Perplexity', 'n']
        with open(outputFile, 'w', newline='') as outputFile:
            writer = csv.writer(outputFile)
            writer.writerow(header)
            writer.writerows(self.hashTablePrediction.values())

        print("\n\n---------------------------------------------------------------------")
        print("Model Output Process: \n")
        print(f"--> You may find prediction results in a CSV file at [{pathOutput}].\n")


def main(argv, predictor=NGramsModel()):
    """
        - The overall control function for the whole model project;
    :param:
        - argv, list, Inputs from the command line, be used to locate data input and output paths;
        - predictor, NGramsModel, one object of the NGramsModel class, which will be edited for processing;
    """

    """ [Initializing] Handling all command parameters entered & Initializing the predictor """
    # Setting for Terminal Running:
    #  If users do not define the model type (optional fourth param), then applying the default model automatically;
    try:
        inputTrainDirectory = argv[1]
        inputDevDirectory = argv[2]
        outputDataFile = argv[3]
        if (len(argv)>4):
            predictor.setModel(modelInput=argv[4])
        else:
            predictor.setModel(modelInput="--unsmoothed")
    # Setting for Manual Running:
    except IndexError:
        inputTrainDirectory = "/Users/den/Desktop/CMPUT 497 - A3/f2021-asn3-Edennnnnnnnnn/data/train/"
        inputDevDirectory = "/Users/den/Desktop/CMPUT 497 - A3/f2021-asn3-Edennnnnnnnnn/data/dev/"
        outputDataFile = "results_dev_unsmoothed.csv"
        predictor.setModel(modelInput=predictor.models[2])
        pass

    """ [Parameters] Applying the heuristic method or the basic method to set the N value for the model """
    # Heuristic Method:
    #  Helps to find the best fitting N value for the model selected by the user;
    predictor.setN_heuristic()
    #  Manual setting function also works;
    # predictor.setN(N=3)

    """ [Input] Adding training and testing/development dataSets to the predictor """
    predictor.modelInput(inputDirectoryPath=inputTrainDirectory, dataType="T")
    predictor.modelInput(inputDirectoryPath=inputDevDirectory, dataType="D")

    """ [Processing] Training the models and make predictions for the test dataset """
    predictor.modelTrain()
    predictor.modelPredict()

    """ [Output] Integrating and writing prediction outputs to the specific file(s) """
    predictor.modelOutput(pathOutput=outputDataFile)

    """ [Analyzing] The Algorithm will evaluate the accuracy of model prediction based on the given prior knowledge """
    predictor.modelAnalyze(printFeedbacks=False)


if __name__ == "__main__":
    main(sys.argv)
