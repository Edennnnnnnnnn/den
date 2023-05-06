import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from knn import KNNModel
from nb import NBModel
from logReg import LogRegModel
from svm import SVMModel
from baseline import BaselineModel


class Classification:
    def __init__(self, cv: int):
        # Data:
        self.datasets_train = None
        self.datasets_test = None
        # Settings:
        self.cv: int = cv
        self.evals: tuple = ('neg_mean_squared_error', 'f1_micro', 'balanced_accuracy')
        self.classifiers: list = [BaselineModel(),
                            LogRegModel(cv=self.cv, cvEval=self.evals),
                            KNNModel(cv=self.cv, cvEval=self.evals),
                            NBModel(cv=self.cv, cvEval=self.evals),
                            SVMModel(cv=self.cv, cvEval=self.evals)]

    def dataSegmentation(self, datasets: tuple):
        #   - Test dataset       (about 20%) --> 0 ~ 114
        #   - Training dataset   (about 80%) --> 115 ~ 455 --> 4-fold Cross Validation:
        # datasets --> (datasets_train, datasets_valid, datasets_test)
        X_train, X_test, y_train, y_test = train_test_split(datasets[0], datasets[1], test_size=0.2, shuffle=True)
        y_train[y_train == 'M'] = 1
        y_train[y_train == 'B'] = 0
        y_test[y_test == 'M'] = 1
        y_test[y_test == 'B'] = 0

        self.datasets_train = [np.delete(X_train, 0, 1).astype(float), np.delete(y_train, 0, 1).astype(int)]
        self.datasets_test = [np.delete(X_test, 0, 1).astype(float), np.delete(y_test, 0, 1).astype(int)]

        print(f"\n----------------------------------------------------------------------------------"
              f"\n* Data loading:"
              f"\n*   datasets_train --> (X: {self.datasets_train[0].shape}, y: {self.datasets_train[1].shape})"
              f"\n*   datasets_test --> (X: {self.datasets_test[0].shape}, y: {self.datasets_test[1].shape})")

    def dataScaling(self):
        scaler = StandardScaler().fit(self.datasets_train[0])
        self.datasets_train[0] = scaler.transform(self.datasets_train[0])
        self.datasets_test[0] = scaler.transform(self.datasets_test[0])

    def dataClassify(self):
        for classifier in self.classifiers:
            if classifier == self.classifiers[0]:
                classifier.classify(datasets_test=self.datasets_test)
                continue
            classifier.training(datasets_train=self.datasets_train)
            classifier.test(datasets_test=self.datasets_test)
            classifier.evaluate()

    @staticmethod
    def dataInput(filename: str) -> tuple:
        labels_hashTable = []
        features_hashTable = []
        with open(filename, newline='') as file:
            contents = csv.reader(file)
            for dataline in contents:
                labels_hashTable.append(dataline[:2])
                features_hashTable.append(dataline[:1] + dataline[2:])
        return np.array(features_hashTable), np.array(labels_hashTable)


def main(classifier=Classification(cv=4)):
    # Data Preprocessing:
    classifier.dataSegmentation(classifier.dataInput(filename='wdbc.csv'))
    classifier.dataScaling()

    # Learning & Classification:
    classifier.dataClassify()


if __name__ == '__main__':
    main()