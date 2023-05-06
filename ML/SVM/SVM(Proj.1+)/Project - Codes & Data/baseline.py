import numpy as np
from sklearn.metrics import f1_score


class BaselineModel:
    def __init__(self):
        # Training Settings:
        self.bestTrainingLoss = np.inf
        self.trainingLosses = []

    def baselineClassifier(self, dataForClassification) -> np.ndarray:
        centroids = []
        for sample in dataForClassification:
            centroid = np.sum(sample) / np.size(sample)
            centroids.append(centroid)
        threshold = np.average(centroids)
        labels = []
        for centroid in centroids:
            if centroid >= threshold:
                labels.append(1)
            else:
                labels.append(0)
        return np.ravel(labels)

    def classify(self, datasets_test: tuple):
        # Data Unpacking:
        X_test = datasets_test[0]
        y_test = np.ravel(datasets_test[1])
        # Baseline Classification:
        y_test_hat = self.baselineClassifier(X_test)

        idx = 0
        matchingCounter = 0
        disCounter = 0
        while idx < np.size(y_test):
            truth = y_test[idx]
            predict = y_test_hat[idx]
            idx += 1
            if truth == predict:
                matchingCounter += 1
            else:
                disCounter += np.abs(truth - predict)
        test_F = f1_score(y_test, y_test_hat)
        test_acc = matchingCounter / np.size(y_test)
        test_loss = disCounter
        print(f"\n----------------------------------------------------------------------------------"
              f"\n* Baselines - The overall prediction analysis is:"
              f"\n*   finalTestFScore --> {test_F}"
              f"\n*   finalTestLoss --> {test_loss}")


