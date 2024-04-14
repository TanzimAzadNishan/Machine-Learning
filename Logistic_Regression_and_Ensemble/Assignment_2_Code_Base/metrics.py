"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np

def count_true_positive(y_true, y_pred):
    true_positive = 0

    # convert dataframe to list
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            true_positive += 1
    
    return true_positive


def count_true_negative(y_true, y_pred):
    true_negative = 0

    # convert dataframe to list
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            true_negative += 1
    
    return true_negative


def count_false_positive(y_true, y_pred):
    false_positive = 0

    # convert dataframe to list
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            false_positive += 1
    
    return false_positive


def count_false_negative(y_true, y_pred):
    false_negative = 0

    # convert dataframe to list
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 0:
            false_negative += 1
    
    return false_negative



def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    # y_true is true labels
    # y_pred is predicted labels

    true_positive = count_true_positive(y_true, y_pred)
    true_negative = count_true_negative(y_true, y_pred)
    false_positive = count_false_positive(y_true, y_pred)
    false_negative = count_false_negative(y_true, y_pred)

    # print(true_positive, true_negative, false_positive, false_negative)

    score = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return score
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    # y_true is true labels
    # y_pred is predicted labels

    true_positive = count_true_positive(y_true, y_pred)
    false_positive = count_false_positive(y_true, y_pred)

    score = true_positive / (true_positive + false_positive)
    return score


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    # y_true is true labels
    # y_pred is predicted labels

    true_positive = count_true_positive(y_true, y_pred)
    false_negative = count_false_negative(y_true, y_pred)
    
    score = true_positive / (true_positive + false_negative)
    return score


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    score = 2 / (1 / precision_score(y_true, y_pred) + 1 / recall_score(y_true, y_pred))
    return score
