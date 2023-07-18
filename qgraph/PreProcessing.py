import matplotlib.pyplot as plt
from pennylane import numpy as np


def standardization(train_set, val_set, bandwidth=0.5):
    """
    Function for standard scaling of one feature of the dataset list
    :param: the_train_set: list of single datas for the training set
    :param: the_val_set: list of single datas for the validation set
    :param: the_bandwidth: value of the bandwidth for the standardization process
    :return: the_y: list of the minimum and maximum values of ground truth
    :return: the_p: list of the minimum and maximum values of momentum
    """

    y_values = [i[1] for i in train_set]
    p_values = [i[0]['p_norm'] for i in train_set]
    y_mean = np.mean(y_values)
    y_std = np.std(y_values)
    p_mean = np.mean(p_values)
    p_std = np.std(p_values)

    assert len(y_values) == len(p_values), "these objects must have the same length"

    standard_scaling(train_set, y_mean, y_std, p_mean, p_std, bandwidth)

    standard_scaling(val_set, y_mean, y_std, p_mean, p_std, bandwidth)

    the_y = np.array([y_mean, y_std])
    the_p = np.array([p_mean, p_std])

    return the_y, the_p


def standard_scaling(the_set, the_y_mean, the_y_std, the_p_mean, the_p_std, the_bandwidth):
    """
    Standard scaling transformation
    :param the_set: list of single datas for the training set
    :param the_y_mean: mean value of the ground truth
    :param the_y_std: standard deviation of the ground truth
    :param the_p_mean: mean value of the momentum
    :param the_p_std: standard deviation of the momentum
    :param the_bandwidth: value of the bandwidth for the standardization process
    :return: None
    """

    set_y = [i[1] for i in the_set]
    set_p_values = [i[0]['p_norm'] for i in the_set]
    for i in range(len(the_set)):
        the_set[i][1] = ((set_y[i] - the_y_mean) / the_y_std) * the_bandwidth
        the_set[i][0]['p_norm'] = ((set_p_values[i] - the_p_mean) / the_p_std) * the_bandwidth


def min_max(train_set, val_set):  # , bandwidth=0.5):
    """
    Function for min_max scaling of one feature of the dataset list
    :param: the_train_set: list of single datas for the training set
    :param: the_val_set: list of single datas for the validation set
    :return: the_y: list of the minimum and maximum values of ground truth
    :return: the_p: list of the minimum and maximum values of momentum
    """

    y_values = [i[1] for i in train_set]
    p_values = [i[0]['p_norm'] for i in train_set]
    y_max = np.max(y_values)
    y_min = np.min(y_values)
    p_max = np.max(p_values)
    p_min = np.min(p_values)

    assert len(y_values) == len(p_values), "these objects must have the same length"

    min_max_scaling(train_set, y_min, y_max, p_min, p_max)

    min_max_scaling(val_set, y_min, y_max, p_min, p_max)

    the_y = np.array([y_min, y_max])
    the_p = np.array([p_min, p_max])

    return the_y, the_p


def min_max_scaling(the_set, the_y_min, the_y_max, the_p_min, the_p_max): #, the_bandwidth):
    """
    Min Max scaling transformation
    :param the_set: list of single datas for the training set
    :param the_y_min: min value of the ground truth
    :param the_y_max: max value of the ground truth
    :param the_p_min: min value of the momentum
    :param the_p_max: max value of the momentum
    :return: None
    """

    set_y = [i[1] for i in the_set]
    set_p_values = [i[0]['p_norm'] for i in the_set]
    for i in range(len(the_set)):
        the_set[i][1] = ((set_y[i] - the_y_min) / (the_y_max - the_y_min))
        the_set[i][0]['p_norm'] = ((set_p_values[i] - the_p_min) / (the_y_max - the_y_min))



