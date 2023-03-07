import pandas as pd
import numpy as np

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class labels
    :return:
    """
    
    df = pd.read_csv('data_banknote_authentication.csv')
    # df = pd.read_excel('Raisin_Dataset.xlsx')


    # print("df = ", df.shape)

    # X = df[['variance', 'skewness', 'curtosis', 'entropy']]

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    # apply standardization to X (mean = 0, std = 1)
    X = (X - X.mean()) / X.std()

    # apply min-max normalization to X (min = 0, max = 1)
    # X = (X - X.min()) / (X.max() - X.min())

    # print("X = ", X.shape)
    # print("y = ", y.shape)

    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """

    # create a dataframe with X and y
    df = pd.DataFrame(X)
    df['class'] = y
    
    if shuffle:
        # When we reset the index, the old index is added as a column, and a new sequential index is used
        # We can use the drop parameter to avoid the old index being added as a column
        df = df.sample(frac=1).reset_index(drop=True)

    X = df.drop('class', axis=1)
    y = df['class']    
    
    split_index = int(test_size * len(X))

    # if test_size is 0.2, then X_train will have 80% of data and X_test will have 20% of data
    
    X_train = X.iloc[:-split_index]
    y_train = y.iloc[:-split_index]
    X_test = X.iloc[-split_index:]
    y_test = y.iloc[-split_index:]

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    
    m = X.shape[0]

    # low = 0, high = m, size = m
    idx = np.random.randint(0, m, m)
    # print("idx = ", idx)

    # use idx to get random samples
    X_sample, y_sample = X.iloc[idx], y.iloc[idx]
    
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample, idx


def shuffle_data(X, y):
    # create dataframe and shuffle the dataset
    df = pd.DataFrame(X)
    df['class'] = y
    df = df.sample(frac=1).reset_index(drop=True)

    shuffled_X = df.drop('class', axis=1)
    shuffled_y = df['class'] 

    return shuffled_X, shuffled_y