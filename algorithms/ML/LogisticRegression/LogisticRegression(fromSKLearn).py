import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class LogRegModel:
    def __init__(self, cv=4, cvEval=('neg_mean_squared_error', 'accuracy')):
        # Training Settings:
        self.bestClassifier = None
        self.bestTrainingLoss = np.inf
        self.trainingLosses = []

        # Cross Validation Settings:
        self.cvEval = cvEval
        self.cv = cv
        self.cvErrors = []
        self.cvAccuracies = []

        # Hyperparameter:
        self.bestC = 0
        self.maxIter = 500
        # todo
        self.hyperParams = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]

    def training(self, datasets_train: tuple):
        # Data Unpacking:
        X_train = datasets_train[0]
        y_train = np.ravel(datasets_train[1])

        # Cross-Validation Process:
        for hyperParam in self.hyperParams:
            classifier = LogisticRegression(C=hyperParam, max_iter=self.maxIter)
            classifier.fit(X_train, y_train)

            # Fitting Evaluation & Scoring:
            MSE = -cross_val_score(classifier, X_train, y_train,
                                   scoring=self.cvEval[0],
                                   cv=self.cv)
            accuracy = cross_val_score(classifier, X_train, y_train,
                                           scoring=self.cvEval[1],
                                           cv=self.cv)
            self.cvErrors.append(np.mean(MSE))
            self.cvAccuracies.append(np.mean(accuracy))

            # Classifier Learning:
            y_train_hat = classifier.predict(X_train)
            trainingLoss = distance.euclidean(y_train, y_train_hat)
            self.trainingLosses.append(trainingLoss)

        self.bestC = self.hyperParams[self.trainingLosses.index(min(self.trainingLosses))]
        self.bestClassifier = LogisticRegression(C=self.bestC, max_iter=self.maxIter)
        self.bestClassifier.fit(X_train, y_train)

    def test(self, datasets_test: tuple):
        # Data Unpacking:
        X_test = datasets_test[0]
        y_test = np.ravel(datasets_test[1])

        # Classifier Testing:
        y_test_hat = self.bestClassifier.predict(X_test)
        testAccuracy = accuracy_score(y_test, y_test_hat)

        # print("* Truth & Prediction values are:")
        # for truth in list(y_test):
            # print(f"*    {truth} - {y_test_hat[list(y_test).index(truth)]}")
        print(f"\n----------------------------------------------------------------------------------"
              f"\n* LogRegModel - The overall prediction analysis is:"
              f"\n*   crossValidationSettings --> {self.cv}-folds, with estimations of {self.cvEval}"
              f"\n*   bestHyperParamFound --> C={self.bestC}"
              f"\n*   finalTestAccuracy --> {testAccuracy}")


    def evaluate(self):
        plt.figure()
        plt.plot(self.hyperParams, self.trainingLosses, color='red')
        plt.xlabel('C-Value (HyperParam)')
        plt.ylabel('Training Loss')
        plt.title('C-Value (HyperParam) vs. Training Loss')
        plt.savefig("LogRegModel_1_hyperParams_vs_trainingLoss.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvErrors, color='green')
        plt.xlabel('C-Value (HyperParam)')
        plt.ylabel('Average MSE (in Cross Validation)')
        plt.title('C-Value (HyperParam) vs. Average MSE')
        plt.savefig("LogRegModel_2_hyperParams_vs_cvAvgErrors.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvAccuracies, color='orange')
        plt.xlabel('C-Value (HyperParam)')
        plt.ylabel('Average Accuracy (in Cross Validation)')
        plt.title('C-Value (HyperParam) vs. Average Accuracy')
        plt.savefig("LogRegModel_3_hyperParams_vs_cvAvgAccuracies.jpg")
