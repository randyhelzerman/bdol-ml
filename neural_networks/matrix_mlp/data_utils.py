import numpy as np
import matplotlib.pyplot as plt
#from py_utils import exit_with_err

"""
Randomly samples p percent of the given data for train and uses the other 1-p
percent for test. Assumes that the data is NxM, where N is the number of
examples and M is the number of features.
"""


def split_train_test(data, target, p=0.7):
    n = data.shape[0]
    num_train = int(np.floor(p * n))
    train_idx = np.random.choice(n, num_train, replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    train_data = data[train_idx, :]
    test_data = data[test_idx, :]
    train_target = target[train_idx]
    test_target = target[test_idx]

    return train_data, test_data, train_target, test_target


"""
Converts integral target values into a NxV indicator matrix, where each row
is an indicator vector of dimension V (if V is the max label). Assumes that
the value "0" is included in the labels.
"""


def integral_to_indicator(integral_target):
    v = int(np.max(integral_target) + 1)
    n = integral_target.shape[0]
    y = np.zeros((n, v))
    for i in range(n):
        y[i, int(integral_target[i])] = 1.0

    return y


def RMSE(yhat, y):
    n = yhat.shape[0]
    return np.sqrt(1.0 / n * np.sum(np.square(yhat - y)))


"""
For each of the variables specified in vars, plot a 2-D plot of the target
vs. the feature.
"""


def plot_regressors(data, target, vars=None, descr=None):
    if vars == None:
        vars = range(0, data.shape[1])

    for i in vars:
        fig = plt.figure()
        plt.scatter(data[:, i], target)
        if descr == None:
            plt.xlabel("Variable {0}".format(i))
        else:
            plt.xlabel("{0}".format(descr[i]))
        plt.ylabel("Target")
        plt.show()
        plt.close(fig)


"""
Given N examples, generate K-folds for cross validation. The indices are
shuffled.

Returns an (N-N/K)xK matrix of training fold indices, and an (N/K)xK matrix of
validation fold indices.
"""


def cross_validation_folds(n, k=5):
    if n % k != 0:
        skip = int(np.floor(float(n)/float(k)))
    else:
        skip = n/k

    ind = np.arange(n)
    np.random.shuffle(ind)

    train_ind = dict()
    val_ind = dict()
    for i in range(k):
        if i == k-1: # Use the rest of the examples
            val = ind[skip*i:]
        else:
            val = ind[skip*i:skip*(i+1)]

        train = np.setdiff1d(ind, val_ind)

        val_ind[i] = val
        train_ind[i] = train

    return train_ind, val_ind

def label_to_bit_vector(labels, nbits):
    bv = np.zeros((labels.shape[0], nbits))
    for i in range(labels.shape[0]):
        bv[i, labels[i]] = 1.0

    return bv

def create_minibatches(data, labels, batch_size, create_bit_vector=False):
    N = data.shape[0]
    if N % batch_size != 0:
        print("Warning in create_minibatches(): Batch size {0} does not " \
              "evenly divide the number of examples {1}.".format(batch_size,
                                                                 N))
    chunked_data = []
    chunked_labels = []
    idx = 0
    while idx + batch_size <= N:
        chunked_data.append(data[idx:idx+batch_size, :])
        if not create_bit_vector:
            chunked_labels.append(labels[idx:idx+batch_size])
        else:
            bv = label_to_bit_vector(labels[idx:idx+batch_size], 10)
            chunked_labels.append(bv)

        idx += batch_size

    return chunked_data, chunked_labels


