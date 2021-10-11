import numpy as np


def mae(y_true, y_pred):
    """
    Simple mean absolute error calculations

    Args:
        y_true: (numpy array) ground truth
        y_pred: (numpy array) predicted values
    Returns:
         (float) mean absolute error
    """
    y_true = y_true.reshape(y_pred.shape)
    return np.mean(np.abs(y_true - y_pred))

    #y_pred = np.array(y_pred + 0.5, dtype=np.int)
    #y_true = y_true.reshape(y_pred.shape)
    #return 1.-np.sum(y_true == y_pred) / len(y_pred)

def accuracy(y_true, y_pred):
    """
    Simple accuracy calculation

    Args:
        y_true: numpy array of 0 and 1's
        y_pred: numpy array of predict sigmoid
    Returns:
        (float) accuracy
    """
    y_pred = y_pred.argmax(-1)
    y_true = y_true.reshape(y_pred.shape)
    #y_pred = y_pred > 0.5
    return np.sum(y_true == y_pred) / len(y_pred)

def accuracy_from_regression(y_true, y_pred):
    y_pred = np.array(y_pred + 0.5, dtype=np.int)
    y_true = y_true.reshape(y_pred.shape)
    return np.sum(y_true == y_pred) / len(y_pred)
