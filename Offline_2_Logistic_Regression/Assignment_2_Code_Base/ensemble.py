import numpy as np

from data_handler import bagging_sampler


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.classifiers = []



    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # temp = []
        
        for i in range(self.n_estimator):
            X_sample, y_sample, idx = bagging_sampler(X, y)
            # temp.append(idx)

            new_classifier = self.base_estimator.fit(X_sample, y_sample, printLoss=False, classifier=i)
            self.classifiers.append(new_classifier)
        
        
        # count = [0 for i in range(1375)]        

        # for i in range(self.n_estimator):
        #     for j in range(1000):
        #         count[temp[i][j]] += 1
        
        
        # for i in range(10):
        #     print("Index: ", i, "Count: ", count[i])




    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """

        m = X.shape[0]
        y_pred = np.zeros(X.shape[0])
        
        for i in range(self.n_estimator):
            y_pred += self.classifiers[i].predict(X)
        
        y_pred_binary = np.zeros(len(y_pred))

        for i in range(m):
            if y_pred[i] > self.n_estimator / 2:
                y_pred_binary[i] = 1
            else:
                y_pred_binary[i] = 0
        
        return y_pred_binary
