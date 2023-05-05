# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt


def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows * ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate((np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate((np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]
    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]
    return X_train, t_train, X_val, t_val, test_data / 256, test_labels

def _createBatches(X, t, batchSize: int, MaxBatch: int):
    # Cutting the dataset to batches for the mini-batch gradient descent processL
    batchDataCollection = []
    currBatchIdx = 0
    while currBatchIdx < MaxBatch:
        currBatchStart = currBatchIdx * batchSize
        currBatchEnd = (currBatchIdx + 1) * batchSize
        X_batch = X[currBatchStart:currBatchEnd, 0:X.shape[1]]
        t_batch = t[currBatchStart:currBatchEnd].reshape(-1, 1)

        batchDataCollection.append((X_batch, t_batch))
        currBatchIdx += 1
    return batchDataCollection


def _crossEntropyFunction(truth, prediction, N=1):
    # CrossEntropy Loss computation:
    # truth --> one-hot expression;
    # prediction --> Softmax returns;
    return (1 / N) * (-np.sum(truth * np.log(prediction)))


def _softmaxFunction(Z):
    # Normalization:
    i = 0
    while i < Z.shape[0]:
        Z[i] -= np.max(Z[i])
        i += 1

    # Computing Softmax:
    array_exp = np.zeros((Z.shape[0], Z.shape[1]))
    i = 0
    while i < Z.shape[0]:
        rowSum = 0
        j = 0
        while j < Z[i].shape[0]:
            rowSum += np.exp(Z[i][j])
            j += 1
        array_exp[i] = np.exp(Z[i]) / rowSum
        i += 1
    return array_exp


def _getOneHotExpression(indexExpress):
    # Getting the one-hot expression for the input matrix of indexExpress:
    N = indexExpress.shape[0]
    oneHotExpress = np.zeros((N, N_class))
    i = 0
    while i < N:
        oneHotExpress[i, indexExpress[i]] = 1
        i += 1
    return oneHotExpress


def train(X_train, t_train, X_val, t_val):
    # Get the num of batches:
    MaxBatch_train = X_train.shape[0] // batch_size
    # Initializing weights and bias:
    W = np.random.random((X_train.shape[1], N_class))
    b = np.random.random(10)

    valid_acc_best = -np.inf
    epoch_best = None
    W_best = None
    b_best = None
    train_losses = []
    valid_accs = []

    epoch = 0
    while epoch < MaxEpoch:
        epochLoss = 0
        # Cutting the dataset to batches for the mini-batch gradient descent processL
        batchDataCollection = _createBatches(X=X_train, t=t_train, batchSize=batch_size, MaxBatch=MaxBatch_train)
        # Epoch Batch-based Training:
        for batch in batchDataCollection:
            # Each batch contains: (X_batch, w_batch, b_batch, t_batch)
            X_batch, t_batch = batch
            t_batch_hat = _softmaxFunction((X_batch @ W) + b)
            # Taking the one-hot expression for further computations:
            t_batch_onehot = _getOneHotExpression(t_batch)
            # Taking the gradient descent processes, record epoch loss:
            W -= alpha * (1 / X_batch.shape[0]) * np.dot(X_batch.T, (t_batch_hat - t_batch_onehot))
            b -= alpha * (1 / X_batch.shape[0]) * np.sum(t_batch_hat - t_batch_onehot)
            epochLoss += _crossEntropyFunction(truth=t_batch_onehot, prediction=t_batch_hat, N=X_batch.shape[0])
        epoch += 1

        # Calling for in-epoch validation process:
        _, _, _, valid_acc = predict(X_val, W, t_val)
        # Counting/recording epoch performance and the optimal values:
        train_losses.append(epochLoss / MaxBatch_train)
        valid_accs.append(valid_acc)
        if valid_acc_best < valid_acc:
            valid_acc_best = valid_acc
            epoch_best = epoch
            W_best = W
            b_best = b
    return valid_acc_best, epoch_best, W_best, b_best, train_losses, valid_accs


def predict(X, W, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K
    # Softmax prediction, then taking the one with the highest value (argmax):
    t_hat = _softmaxFunction(X @ W)
    t_hat_argmax = np.argmax(t_hat, axis=1).reshape(-1, 1)
    # Taking the one-hot expression for further computations:
    t_onehot = _getOneHotExpression(t)
    # Computing the CrossEntropyLoss, and the prediction accuracy:
    loss = _crossEntropyFunction(truth=t_onehot, prediction=t_hat, N=X.shape[0])
    acc = evaluate(t_hat_argmax, t)
    return t, t_hat, loss, acc


def evaluate(t, t_hat):
    """
    Calculate accuracy,
    """
    acc = (t == t_hat).sum() / t.shape[0]
    return acc


def plot(train_losses, valid_accs):
    plt.figure()
    plt.plot(np.arange(1, MaxEpoch + 1), train_losses)
    plt.xlabel("#Epoch")
    plt.ylabel("Training Losses")
    plt.title("#Epoch vs. Training Losses")
    plt.savefig(f"plot1_#Epoch_vs_TrainingLosses_{batch_size}.jpg")
    plt.cla()

    plt.figure()
    plt.plot(np.arange(1, MaxEpoch + 1), valid_accs)
    plt.xlabel("#Epoch")
    plt.ylabel("ValidationAccuracies")
    plt.title("#Epoch vs. ValidationAccuracies")
    plt.savefig(f"plot2_#Epoch_vs_ValidationAccuracies_{batch_size}.jpg")


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()
print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 10
alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay

valid_acc_best, epoch_best, W_best, b_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)
_, _, _, acc_test = predict(X_test, W_best, t_test)


# TODO: report 3 number, plot 2 curves
print(f"\n#epoch yields the best validation performance is {epoch_best}:")
print(f"  * with validation performance (accuracy) of --> {valid_acc_best}")
print(f"  * with test performance (accuracy) of --> {acc_test}")

plot(train_losses, valid_accs)
