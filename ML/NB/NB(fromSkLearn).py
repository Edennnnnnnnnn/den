import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class NBModel:
    def __init__(self, cv=4, cvEval=('neg_mean_squared_error', 'accuracy')):
        # Training Settings:
        self.bestClassifier = None
        self.bestTrainingLoss = np.inf
        self.trainingLosses = []
        #self.scaler = None

        # Cross Validation Settings:
        self.cvEval = cvEval
        self.cv = cv
        self.cvErrors = []
        self.cvAccuracies = []

        # Hyperparameter:
        self.bestAlpha = 0
        # todo
        self.hyperParams = [0.1, 0.5, 1, 5, 10, 50, 100, 500]

    def training(self, datasets_train: tuple):
        # Data Unpacking:
        X_train = datasets_train[0]
        y_train = np.ravel(datasets_train[1])
        #self.scaler = StandardScaler().fit(X_train)
        #X_train = self.scaler.transform(X_train)

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
            self.cvErrors.append(np.mean(MSE))
            self.cvAccuracies.append(np.mean(accuracy))

            # Classifier Learning:
            #classifier.fit(X_train, y_train)
            y_train_hat = classifier.predict(X_train)
            trainingLoss = distance.euclidean(y_train, y_train_hat)
            self.trainingLosses.append(trainingLoss)

        self.bestAlpha = self.hyperParams[self.trainingLosses.index(min(self.trainingLosses))]
        self.bestClassifier = BernoulliNB(alpha=self.bestAlpha)
        self.bestClassifier.fit(X_train, y_train)

    def test(self, datasets_test: tuple):
        # Data Unpacking:
        X_test = datasets_test[0]
        y_test = np.ravel(datasets_test[1])
        #X_test = self.scaler.transform(X_test)

        # Classifier Testing:
        y_test_hat = self.bestClassifier.predict(X_test)
        testAccuracy = accuracy_score(y_test, y_test_hat)

        # print("* Truth & Prediction values are:")
        # for truth in list(y_test):
            # print(f"*    {truth}   -   {y_test_hat[list(y_test).index(truth)]}")
        print(f"\n----------------------------------------------------------------------------------"
              f"\n* NBModel - The overall prediction analysis is:"
              f"\n*   crossValidationSettings --> {self.cv}-folds, with estimations of {self.cvEval}"
              f"\n*   bestHyperParamFound --> α={self.bestAlpha}"
              f"\n*   finalTestAccuracy --> {testAccuracy}")


    def evaluate(self):
        plt.figure()
        plt.plot(self.hyperParams, self.trainingLosses, color='red')
        plt.xlabel('α-Value (HyperParam)')
        plt.ylabel('Training Loss')
        plt.title('α-Value (HyperParam) vs. Training Loss')
        plt.savefig("NBModel_1_hyperParams_vs_trainingLoss.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvErrors, color='green')
        plt.xlabel('α-Value (HyperParam)')
        plt.ylabel('Average MSE (in Cross Validation)')
        plt.title('α-Value  Negative Mean Squared Error')
        plt.savefig("NBModel_2_hyperParams_vs_cvAvgErrors.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvAccuracies, color='orange')
        plt.xlabel('α-Value (HyperParam)')
        plt.ylabel('Average Accuracy (in Cross Validation)')
        plt.title('α-Value  (HyperParam) vs. Average Accuracy')
        plt.savefig("NBModel_3_hyperParams_vs_cvAvgAccuracies.jpg")
