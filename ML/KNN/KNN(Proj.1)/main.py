import torch
from torchvision import datasets, transforms
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat

from knn import knn


def compute_base_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


def compute_runtime_factor(runtime, min_thres, max_thres):
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
    predicted_y_test = algorithm(x_train, y_train, x_test, n_classes, device)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start

    correct_predict = (y_test
                       == predicted_y_test).astype(np.int32).sum()
    accuracy = float(correct_predict) / len(y_test)

    print('Correct Predictions: {}/{} total \tAccuracy: {:5f} \tTime: {:2f}'.format(correct_predict,
                                                                                    len(y_test), accuracy, run_time))
    return correct_predict, accuracy, run_time


def main():
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

    # You may want to use a smaller training set to save time when debugging
    # i.e.: Put something like:
    # (x_train, y_train) = (x_train[:5000], y_train[:5000])

    # For this assignment, we only test on the first 1000 samples of the test set
    (x_valid, y_valid) = (x_valid[:1000], y_valid[:1000])

    print("Dimension of dataset: ")
    print("Train:", x_train.shape, y_train.shape, "\nTest:", x_valid.shape, y_valid.shape)

    (correct_predict, accuracy, run_time) = run(knn, x_train, y_train, x_valid, y_valid, n_classes, device)
    base_score = compute_base_score(accuracy, min_acc_thres, max_acc_thres)

    runtime_factor = compute_runtime_factor(run_time, min_runtime_thres, max_runtime_thres)

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


if __name__ == "__main__":
    main()
