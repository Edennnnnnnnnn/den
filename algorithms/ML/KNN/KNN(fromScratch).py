"""
By Eden Zhou
Sep. 15, 2022
"""

import torch
import numpy
import timeit
from collections import OrderedDict
from pprint import pformat
from torchvision import datasets, transforms


class KNNModel:
    def __init__(self):
        """ Class Variables Initialized"""
        self.k: int = 3
        self.device: str = ''

        self.x_train: torch = None
        self.y_train: torch = None
        self.x_test: torch = None
        self.y_test: numpy = None

    def __str__(self):
        print("""
        # COPYRIGHT
        ----------------------------------------------------------------------------
        By Eden Zhou
        Sep. 15, 2022


        # MODEL & ALGORITHM
        ----------------------------------------------------------------------------
        * KNNModel:
        [Supervised Learning]
        --> K Nearest Neighbors (KNN) algorithm; 
        
    
         # DATA & HELPS
        ----------------------------------------------------------------------------
        * Possible dataset can be found in `datasets.MNIST` ;
        * SampleCalling Function provided;
        * More evaluation parts can be found in the last commented lines of this file;
    

        # FUNCTIONS
        ----------------------------------------------------------------------------
        --> setDevice(self, device) -> None:
            This function accepts the device info, and applies it to the model; Can be used for GPU accelerations;
        :param:
            - self;
            - device, str, the local device info;
            
        ----------------------------------------------------------------------------
        --> setK(self, K) -> None:
            This function accepts the user-defined K value, and applies it to the model;
        :param:
            - self;
            - K, int, the K value for the K-NN model;
            
        ----------------------------------------------------------------------------
        --> modelInput_training(self, x_train, y_train) -> None:
            This function accepts the input training data/label sets (x & y) and loading it into the model;
        :param:
            - self;
            - x_train, the training data (x) been loaded into the model;
            - y_train, the training labels (y) been loaded into the model;
        
        ----------------------------------------------------------------------------
        --> modelInput_predicting(self, x_test) -> None:
            This function accepts the input testing data set and loading it into the model;
        :param:
            - self;
            - x_test, the testing data (x) been loaded into the model;
        
        ----------------------------------------------------------------------------
        --> tensorTransfer(self, inputArray, device=None) -> torch:
            This function accepts numpy arrays and transfering them to be pytorch tensors for some further operations;
        :param:
            - self;
            - inputArray, 2d array;
            - device, str, presents the info of the local machine;
        :return:
            - outputTensor, 2d tensor;
        
        ----------------------------------------------------------------------------
        --> modelRun(self) -> list:
            This method runs K-NN algorithms to make predictions for the input test data based on the prior
        knowledge learned from training data set; Due to the features of K-NN, this function contains both the training
        pt. and the prediction pt;
        :param:
            - self;
        :return:
            - y_test, 2d list, predicted y_test which is a 2d vector, corresponding to the size of the input x_test;
        
        ----------------------------------------------------------------------------
        --> modelOutput(self) -> None:
            This method helps to print the prediction results to the command line window, further operations may be
        added to output the relevant prediction results to specific file paths;
        :param:
            - self;
            
        """)

    def setDevice(self, device) -> None:
        """
            This function accepts the device info, and applies it to the model; Can be used for GPU accelerations;
        :param:
            - self;
            - device, str, the local device info;
        """
        self.device = device

    def setK(self, K) -> None:
        """
            This function accepts the user-defined K value, and applies it to the model;
        :param:
            - self;
            - K, int, the K value for the K-NN model;
        """
        self.k = K

    def modelInput_training(self, x_train, y_train) -> None:
        """
            This function accepts the input training data/label sets (x & y) and loading it into the model;
        :param:
            - self;
            - x_train, the training data (x) been loaded into the model;
            - y_train, the training labels (y) been loaded into the model;
        """
        self.x_train = self.tensorTransfer(x_train)
        self.y_train = self.tensorTransfer(y_train)

    def modelInput_predicting(self, x_test) -> None:
        """
            This function accepts the input testing data set and loading it into the model;
        :param:
            - self;
            - x_test, the testing data (x) been loaded into the model;
        """
        self.x_test = self.tensorTransfer(x_test)

    def tensorTransfer(self, inputArray, device=None) -> torch:
        """
            This function accepts numpy arrays and transfering them to be pytorch tensors for some further operations;
        :param:
            - self;
            - inputArray, 2d array;
            - device, str, presents the info of the local machine;
        :return:
            - outputTensor, 2d tensor;
        """
        if device is None:
            device = self.device
        outputTensor = torch.tensor(inputArray, dtype=torch.float, device=device)
        return outputTensor

    def modelRun(self) -> list:
        """
            This method runs K-NN algorithms to make predictions for the input test data based on the prior
        knowledge learned from training data set; Due to the features of K-NN, this function contains both the training
        pt. and the prediction pt;
        :param:
            - self;
        :return:
            - y_test, 2d list, predicted y_test which is a 2d vector, corresponding to the size of the input x_test;
        """
        y_test = []
        testLength = len(self.x_test)
        for i in range(testLength):
            # Apply Torch Broadcasting to each image to fit the shape of the training set provided; Compute the
            #  Euclidean distance b/w each "neighbor" and the target test image (row);
            x_test_i_tensor_bc = torch.broadcast_to(self.x_test[i], (self.x_train.size(dim=0), self.x_train.size(dim=1)))
            rawDist = (x_test_i_tensor_bc - self.x_train) ** 2
            distances = rawDist.sum(axis=1) ** 0.5

            # Based on the k setting, get k nearest neighbors (from the trainSet) around the target test image;
            knnIdx = distances.topk(self.k, largest=False)[1]

            # Add the prediction of knn (predictions in y_train) to the list predictor; Find the one that appeared most
            #  frequently, convert it back to numpy array and add it to y_test;
            predictor = []
            for idx in knnIdx:
                predictor.append(self.y_train[idx])
            y_test.append(max(predictor, key=predictor.count).cpu().numpy())

        self.y_test = y_test
        return y_test

    def modelOutput(self) -> None:
        """
            This method helps to print the prediction results to the command line window, further operations may be
        added to output the relevant prediction results to specific file paths;
        :param:
            - self;
        """
        print("Model Prediction Results: ")
        for case in self.y_test:
            print(case)


def sampleMain(predictor=KNNModel()):
    predictor.setK(3)
    predictor.setDevice("DEVICE_NAME")

    predictor.modelInput_training(...)
    predictor.modelInput_predicting(...)

    predictor.modelRun()




"""
def modelAnalyze_compute_base_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) *100
    return base_score


def modelAnalyze_compute_runtime_factor(runtime, min_thres, max_thres):
    if runtime <= min_thres:
        score_factor = 1
    elif runtime >= max_thres:
        score_factor = 0
    else:
        score_factor = 0.5

    return score_factor


def run(algorithm, x_train, y_train, x_test, y_test, n_classes, device):
    print('Running...')

    if device != 'cpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Training on CPU')

    start = timeit.default_timer()
    np.random.seed(0)
    predicted_y_test = KNNModel.knn(x_train, y_train, x_test, n_classes, device)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start

    correct_predict = (y_test
                       == predicted_y_test).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)

    print('Correct Predictions: {}/{} total \tAccuracy: {:5f} \tTime: {:2f}'.format(correct_predict,
                                                                                    len(y_test), accuracy,
                                                                                    run_time))
    return correct_predict, accuracy, run_time


def SampleMain():
    min_acc_thres = 0.84
    max_acc_thres = 0.94

    min_runtime_thres = 12
    max_runtime_thres = 24

    n_classes = 10
    # change to cpu to run on CPU
    device = 'gpu'

    mnist_train = datasets.MNIST('data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.Normalize((0.1307,), (0.3081,)),
                                 ])
                                 )
    mnist_test = datasets.MNIST('data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                ])
                                )
    # convert pytorch tensors to numpy arrays
    (x_train, y_train) = (mnist_train.data.cpu().numpy(), mnist_train.targets.cpu().numpy())
    (x_valid, y_valid) = (mnist_test.data.cpu().numpy(), mnist_test.targets.cpu().numpy())

    # flatten 28x28 images into 784 sized vectors
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)

    (x_valid, y_valid) = (x_valid[:1000], y_valid[:1000])

    print("Dimension of dataset: ")
    print("Train:", x_train.shape, y_train.shape, "\nTest:", x_valid.shape, y_valid.shape)

    (correct_predict, accuracy, run_time) = run(KNNModel(), x_train, y_train, x_valid, y_valid, n_classes, device)
    base_score = modelAnalyze_compute_base_score(accuracy, min_acc_thres, max_acc_thres)
    runtime_factor = modelAnalyze_compute_runtime_factor(run_time, min_runtime_thres, max_runtime_thres)
    overall_score = base_score * runtime_factor
    
    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy,
                         run_time=run_time,
                         base_score=base_score,
                         overall_score=overall_score
                         )

    with open('result.txt', 'w') as f:
        f.writelines(pformat(result, indent=4))

    print(pformat(result, indent=4))

"""