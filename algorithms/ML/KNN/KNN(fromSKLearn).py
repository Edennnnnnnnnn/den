import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class KNNModel:
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
        self.bestK = 0
        self.hyperParams = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    def training(self, datasets_train: tuple):
        # Data Unpacking:
        X_train = datasets_train[0]
        y_train = np.ravel(datasets_train[1])

        # Cross-Validation Process:
        for hyperParam in self.hyperParams:
            classifier = KNeighborsClassifier(n_neighbors=hyperParam)
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

        self.bestK = self.hyperParams[self.trainingLosses.index(min(self.trainingLosses))]
        self.bestClassifier = KNeighborsClassifier(n_neighbors=self.bestK)
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
              f"\n* KNNModel - The overall prediction analysis is:"
              f"\n*   crossValidationSettings --> {self.cv}-folds, with estimations of {self.cvEval}"
              f"\n*   bestHyperParamFound --> k={self.bestK} (neighbors)"
              f"\n*   finalTestAccuracy --> {testAccuracy}")


    def evaluate(self):
        plt.figure()
        plt.plot(self.hyperParams, self.trainingLosses, color='red')
        plt.xlabel('#Neighbor (HyperParam)')
        plt.ylabel('Training Loss')
        plt.title('k-value (HyperParam) vs. Training Loss')
        plt.savefig("KNNModel_1_hyperParams_vs_trainingLoss.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvErrors, color='green')
        plt.xlabel('#Neighbor (HyperParam)')
        plt.ylabel('Average MSE (in Cross Validation)')
        plt.title('k-value (HyperParam) vs. Average MSE')
        plt.savefig("KNNModel_2_hyperParams_vs_cvAvgErrors.jpg")

        plt.figure()
        plt.plot(self.hyperParams, self.cvAccuracies, color='orange')
        plt.xlabel('#Neighbor (HyperParam)')
        plt.ylabel('Average Accuracy (in Cross Validation)')
        plt.title('k-value (HyperParam) vs. Average Accuracy')
        plt.savefig("KNNModel_3_hyperParams_vs_cvAvgAccuracies.jpg")

