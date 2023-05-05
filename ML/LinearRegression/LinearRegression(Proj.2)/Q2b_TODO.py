#!/usr/bin/env python3

# import sklearn.datasets as datasets
import numpy as np
import pickle as pickle

from matplotlib import pyplot as plt


def predict(X, w, y=None):
    # Y = Xw; Y --> (N_batch, 1), X --> (N_batch * (d+1)), w --> ((d+1) * 1)
    y_origin = y * std_y + mean_y
    # y_hat_norm --> y_hat
    # y_hat --> y_hat_origin
    y_hat = X @ w
    #y_hat_norm = (y_hat - np.mean(y_hat)) / np.std(y_hat)
    y_hat_origin = y_hat * std_y + mean_y

    # Loss, J-Value:
    loss = (1 / (2 * y.shape[0])) * np.sum(np.abs(y - y_hat) ** 2)
    # Risk/Error, E-Value:
    risk = (1 / y_origin.shape[0]) * np.sum(np.abs(y_origin - y_hat_origin))
    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val, decayTerm) -> tuple:
    # Y = Xw; Y --> (N_train, 1), X --> (N_train * (d+1)), w --> ((d+1) * 1)
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: ((d+1) * 1)

    losses_train = []
    risks_val = []
    w_best = None
    risk_best = 10000
    epoch_best = 0
    for epoch in range(MaxIter):
        loss_this_epoch = 0
        for batch in range(int(np.ceil(N_train / batch_size))):
            X_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
            y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent
            w = w - alpha * ((1 / len(y_batch)) * np.dot(np.transpose(X_batch), (y_hat_batch - y_batch)) + decayTerm * w)

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        batchNum = np.ceil(N_train / batch_size)
        losses_train.append(loss_this_epoch / batchNum)

        # 2. Perform validation on the validation set by the risk
        _, _, risk_val = predict(X_val, w, y_val)
        risks_val.append(risk_val)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            risk_best = risk_val
            epoch_best = epoch
            w_best = w

    # Return some variables as needed
    return risk_best, w_best, epoch_best, losses_train, risks_val



############################
# Main code starts here
############################
# Load data. This is the only allowed API call from sklearn
#X, y = datasets.load_boston(return_X_y=True)
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

X = np.hstack((X, X ** 2))
y = y.reshape([-1, 1])
# X: (N * d)
# y: (N * 1)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: (N * (d+1))

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - np.mean(y)) / np.std(y)

print(X.shape, y.shape) # It's always helpful to print the shape of a variable

# Randomly shuffle the data;
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# Setting:
alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay
decayTerms = [3, 1, 0.3, 0.1, 0.03, 0.01]

# Take the training and validation process;
epoch_bests = []
risk_bests = []
risk_tests = []
for decayTerm in decayTerms:
    risk_best, w_best, epoch_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val, decayTerm)

    # Perform test by the weights yielding the best validation performance;
    y_hat_test, loss_test, risk_test = predict(X_test, w_best, y_test)

    # Report test results ;
    print(f"\n\nCurrent DecayTerm --> {decayTerm}")
    print(f"The number of epoch that yields the best validation performance is {epoch_best + 1}")
    print(f"The validation performance (risk) in that epoch is {risk_best}")
    print(f"The test performance (risk) in that epoch is {risk_test}")

    epoch_bests.append(epoch_best)
    risk_bests.append(risk_best)
    risk_tests.append(risk_test)

    # Report numbers and draw plots as required;
    plt.figure(figsize=(12, 6))
    epochNum = np.arange(1, len(losses_train) + 1)
    plt.plot(epochNum, losses_train, color="blue")
    plt.title("Learning Curve of The Training Loss")
    plt.xlabel('Num of Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('Question 2b-1' + f'_{decayTerm}' + '.jpg')
    plt.show()

    plt.figure(figsize=(12, 6))
    epochNum = np.arange(1, len(risks_val) + 1)
    plt.plot(epochNum, risks_val, color="green")
    plt.title("Learning Curve of The Validation Risk")
    plt.xlabel('Num of Epochs')
    plt.ylabel('Risk')
    plt.tight_layout()
    plt.savefig('Question 2b-2' + f'_{decayTerm}' + '.jpg')
    plt.show()


plt.figure()
plt.plot(decayTerms, epoch_bests, color="red")
plt.xlabel('decayTerms')
plt.ylabel('epoch_bests')
plt.tight_layout()
plt.savefig('Question 2b-3 epoch_bests' + '.jpg')
plt.show()

plt.figure()
plt.plot(decayTerms, risk_bests, color="yellow")
plt.plot(decayTerms, risk_tests, color="black")
plt.xlabel('decayTerms')
plt.ylabel('risks')
plt.tight_layout()
plt.savefig('Question 2b-3 risks' + '.jpg')
plt.show()

bestDecayTerm_index = risk_bests.index(min(risk_bests))
print(f"\n\n---------------------------------------\nBest DecayTerm --> {decayTerms[bestDecayTerm_index]}")