from pennylane import numpy as np


def standardization(the_train_set, the_val_set, the_bandwidth=1.):
    """
    Function for standard scaling of one feature of the dataset list
    :param: the_train_set: list of single datas for the training set
    :param: the_val_set: list of single datas for the validation set
    :param: the_test_set : list of single datas for the test set
    :param: the_bandwidth: value of the bandwidth for the standardization process
    """

    y_values = [i[1] for i in the_train_set]
    p_values = [i[0]['p_norm'] for i in the_train_set]
    the_y_mean = np.mean(y_values)
    the_y_std = np.std(y_values)
    the_p_mean = np.mean(p_values)
    the_p_std = np.std(p_values)

    assert len(y_values) == len(p_values), "these objects must have the same length"

    standard_scaling(the_train_set, the_y_mean, the_y_std, the_p_mean, the_p_std, the_bandwidth)

    standard_scaling(the_val_set, the_y_mean, the_y_std, the_p_mean, the_p_std, the_bandwidth)

    the_y = np.array([the_y_mean, the_y_std])
    the_p = np.array([the_p_mean, the_p_std])

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

