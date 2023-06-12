import matplotlib.pyplot as plt
from pennylane import numpy as np

## HAVE TO CONTROL WHY THE STANDARDIZATION FUNCTION DOESN'T WORK!!!


def standardization(the_train_set, the_val_set, the_test_set, the_bandwidth=1.):
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

    for i in range(len(y_values)):
        the_train_set[i][1] = ((y_values[i] - the_y_mean) / the_y_std) * the_bandwidth
        the_train_set[i][0]['p_norm'] = ((p_values[i] - the_p_mean)/the_p_std)*the_bandwidth
    train_y_norm = [i[1] for i in the_train_set]
    train_angle = [i[0]['theta'] for i in the_train_set]
    plt.plot(train_angle, y_values, 'ro')
    plt.plot(train_angle, train_y_norm, 'bo')
    plt.show()

    val_y = [i[1] for i in the_val_set]
    val_p_values = [i[0]['p_norm'] for i in the_val_set]
    for i in range(len(the_val_set)):
        the_val_set[i][1] = ((val_y[i] - the_y_mean)/the_y_std)*the_bandwidth
        the_val_set[i][0]['p_norm'] = ((val_p_values[i] - the_p_mean) / the_p_std) * the_bandwidth

    val_y_norm = [i[1] for i in the_val_set]
    val_angle = [i[0]['theta'] for i in the_val_set]
    plt.plot(val_angle, val_y, 'ro')
    plt.plot(val_angle, val_y_norm, 'bo')
    plt.show()

    test_y = [i[1] for i in the_test_set]
    test_p_values = [i[0]['p_norm'] for i in the_test_set]
    for i in range(len(the_val_set)):
        the_test_set[i][1] = ((test_y[i] - the_y_mean) / the_y_std) * the_bandwidth
        the_test_set[i][0]['p_norm'] = ((test_p_values[i] - the_p_mean) / the_p_std) * the_bandwidth

    test_y_norm = [i[1] for i in the_test_set]
    test_angle = [i[0]['theta'] for i in the_test_set]
    plt.plot(test_angle, test_y, 'ro')
    plt.plot(test_angle, test_y_norm, 'bo')
    plt.show()

    the_y = [the_y_mean, the_y_std]
    the_p = [the_p_mean, the_p_std]

    return the_y, the_p
