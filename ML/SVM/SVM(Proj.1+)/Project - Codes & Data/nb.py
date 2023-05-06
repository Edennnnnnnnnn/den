import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score


class NBModel:
    def __init__(self, cv: int, cvEval: tuple):
        # Training Settings:
        self.bestClassifier = None
        self.bestTrainingLoss = np.inf
        self.trainingLosses = []

        # Cross Validation Settings:
        self.cvEval: tuple = cvEval
        self.cv: int = cv
        self.cvErrors: list = []
        self.cvFValues: list = []
        self.cvBalAcc:list = []

        # Hyperparameter:
        self.bestAlpha = 0
        self.hyperParams = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

    def training(self, datasets_train: tuple):
        # Data Unpacking:
        X_train = datasets_train[0]
        y_train = np.ravel(datasets_train[1])

        # Cross-Validation Process:
        for hyperParam in self.hyperParams:
            classifier = BernoulliNB(alpha=hyperParam)
            classifier.fit(X_train, y_train)

            # Fitting Evaluation & Scoring:
            MSE = -cross_val_score(classifier, X_train, y_train,
                                   scoring=self.cvEval[0],
                                   cv=self.cv)
            accuracy = cross_val_score(classifier, X_train, y_train,
                                           scoring=self.cvEval[1],
                                           cv=self.cv)
            balancedAccuracies = cross_val_score(classifier, X_train, y_train,
                                                 scoring=self.cvEval[2],
                                                 cv=self.cv)
            self.cvErrors.append(np.mean(MSE))
            self.cvFValues.append(np.mean(accuracy))
            self.cvBalAcc.append(np.mean(balancedAccuracies))

        self.bestAlpha = self.hyperParams[self.cvFValues.index(max(self.cvFValues))]
        self.bestClassifier = BernoulliNB(alpha=self.bestAlpha)
        self.bestClassifier.fit(X_train, y_train)

    def test(self, datasets_test: tuple):
        # Data Unpacking:
        X_test = datasets_test[0]
        y_test = np.ravel(datasets_test[1])
        #X_test = self.scaler.transform(X_test)

        # Classifier Testing:
        y_test_hat = self.bestClassifier.predict(X_test)
        testFScore = f1_score(y_test, y_test_hat)

        print(f"\n----------------------------------------------------------------------------------"
              f"\n* NBModel - The overall prediction analysis is:"
              f"\n*   crossValidationSettings --> {self.cv}-folds, with estimations of {self.cvEval}"
              f"\n*   bestHyperParamFound --> α={self.bestAlpha}"
              f"\n*   finalTestFScore --> {testFScore}")

    def evaluate(self):
        plt.figure()
        plt.plot(self.hyperParams, self.cvBalAcc, color='red')
        plt.xlabel('α-Value (HyperParam)')
        plt.ylabel('Mean of Balanced Accuracy (in Cross Validation)')
        plt.title('α-Value (HyperParam) vs. Mean of Balanced Accuracy')
        plt.savefig("NBModel_1_hyperParams_vs_balanced_accuracy.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvErrors, color='green')
        plt.xlabel('α-Value (HyperParam)')
        plt.ylabel('Average MSE (in Cross Validation)')
        plt.title('α-Value (HyperParam) vs. Average MSE')
        plt.savefig("NBModel_2_hyperParams_vs_cvAvgErrors.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvFValues, color='orange')
        plt.xlabel('α-Value (HyperParam)')
        plt.ylabel('Mean of F1-values (in Cross Validation)')
        plt.title('α-Value (HyperParam) vs. Mean of F1-Values')
        plt.savefig("NBModel_3_hyperParams_vs_cvFValues.jpg")
