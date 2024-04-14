import numpy as np
from data_handler import shuffle_data

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """

        # params is a dictionary with attributes learning_rate, max_iterations, threshold, lambda, batch_size, verbose
        self.learning_rate = params['learning_rate']
        self.max_iterations = params['max_iterations']
        self.threshold = params['threshold']
        self.lambda_ = params['lambda']
        self.batch_size = params['batch_size']
        self.verbose = params['verbose']


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def compute_log_loss(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        log_loss = 0.0

        # X is a dataframe of shape (m, n). convert it to a numpy array of shape (m, n)
        X = X.to_numpy()
        y = y.to_numpy()

        for i in range(m):
            z = X[i].dot(self.weights) + self.bias
            y_pred = self.sigmoid(z)
            log_loss += (-y[i] * np.log(y_pred) - (1 - y[i]) * np.log(1 - y_pred))

        
        log_loss /= m
        # add regularization term
        log_loss += (1/(2*n)) * self.lambda_ * np.sum(self.weights ** 2)
        
        return log_loss


    def gradient_descent(self, X, y):
        m = X.shape[0]
        n = X.shape[1]

        # Apply the sigmoid function element-wise to the dot product of X and the weights
        y_pred = self.sigmoid(X.dot(self.weights) + self.bias)

        # Calculate the gradient of the loss function
        dw = (1 / m) * X.T.dot(y_pred - y) + (1/n) * self.lambda_ * self.weights
        db = (1 / m) * np.sum(y_pred - y)

        # Update the weights and the bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db


    def fit(self, X, y, printLoss=True, classifier=None):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # Initialize the weights and the bias to zero
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # print('batch size', self.batch_size)
        
        for i in range(self.max_iterations):

            # shuffle the dataset for each epoch
            if i != 0:
                X, y = shuffle_data(X, y)
            
            # loop over mini batches
            for j in range(0, X.shape[0], self.batch_size):
                X_batch = X.iloc[j:j+self.batch_size]
                y_batch = y.iloc[j:j+self.batch_size]

                # call gradient_descent function
                self.gradient_descent(X_batch, y_batch)


            if printLoss and self.verbose == True and i % 100 == 0:
                print(f'Loss after iteration {i} : {self.compute_log_loss(X, y)}')
        
        if classifier is not None and self.verbose == True:
            print('Classifier', classifier, end=': ')
        
        if self.verbose == True:
            print(f'Loss after iteration {self.max_iterations} : {self.compute_log_loss(X, y)}')

        return self

    
    
    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """

        y_pred = self.sigmoid(X.dot(self.weights) + self.bias)

        # convert y_pred to a numpy array
        y_pred = y_pred.to_numpy()
        y_pred_binary = np.zeros(len(y_pred))

        # Convert the probabilities to binary values
        for i in range(len(y_pred)):
            if y_pred[i] > self.threshold:
                y_pred_binary[i] = 1
            else:
                y_pred_binary[i] = 0

        
        return y_pred_binary
