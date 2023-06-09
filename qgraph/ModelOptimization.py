import time
from pennylane import numpy as np
import torch
from torch import optim
from torch_geometric.utils import to_networkx
from qgraph import expect_value
import matplotlib.pyplot as plt


def predict(the_dataset, the_weights, the_n_layers, the_choice: str):
    """
    :param the_dataset: either validation or test set, list of datas
    :param the_weights: array of trained parameters
    :param the_n_layers: number of layers of the ansatz (depth of the circuit)
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: list of predictions
    """
    return [expect_value(element[0], the_n_layers, the_weights, the_choice) for element in the_dataset]


def get_mse(predictions, ground_truth):
    """
    :param predictions: list of predictions of the QGNN
    :param  ground_truth: list of true labels
    :return: mse: sum of squared errors
    """
    n = len(predictions)
    assert len(ground_truth) == n, "The number of predictions and true labels is not equal"
    return sum([(predictions[i] - torch.tensor(ground_truth[i])) ** 2 for i in range(n)])


"""
here the_training_set must be a list tuple (graph, output)!!!!!
"""


def train_qgnn(the_training_loader, the_validation_loader, the_init_weights, the_n_epochs, the_n_layers=3,
               the_choice: str = 'parametrized'):
    """
    :param  the_training_loader: DataLoader object of the training set
    :param the_validation_loader: DataLoader object of the validation set
    :param the_n_layers: numbers of layers of the quantum circuit
    :param the_init_weights: parameters to insert in the quantum circuit
    :param  the_n_epochs: number of epochs of the training process
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: the_weights: list of the final weights after the training
    """

    opt = optim.Adam([the_init_weights], lr=1e-2)  # initialization of the optimizer to use
    the_weights = the_init_weights
    epoch_loss = []
    validation_loss = []

    for epoch in range(the_n_epochs):

        costs = []
        starting_time = time.time()

        for _, item in enumerate(the_training_loader):
            mini_batch = []
            # converting for each batch any DataLoader item into a list of tuples of networkx graph
            # object and the corresponding output
            mini_batch = [(to_networkx(data=item[0][i], node_attrs=['state', 'p_norm', 'theta'],
                                       edge_attrs=['mass', 'spin', 'charge'], to_undirected=True),
                           item[1][i]) for i in range(len(item[0]))]

            def opt_func():  # defining an optimization function for the training of the model
                mini_batch_predictions = predict(mini_batch, the_weights, the_n_layers, the_choice)
                mini_batch_truth = [element[1] for element in mini_batch]
                loss = get_mse(mini_batch_predictions, np.array(mini_batch_truth, dtype=object))
                costs.append(loss.item())
                loss.backward()
                return loss

            opt.zero_grad()
            opt.step(opt_func)

        ending_time = time.time()
        elapsed = ending_time - starting_time

        training_loss = np.mean(costs)
        epoch_loss.append(training_loss)

        the_val = validation_qgnn(the_validation_loader, the_weights, the_choice, the_n_layers)
        validation_loss.append(the_val)

        if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-3:
            the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            break

        if epoch % 5 == 0:
            res = [epoch, training_loss, elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | Elapsed Time per Epoch: {:3f}".format(*res))

    if the_choice == 'unparametrized':
        the_filename1 = '../data/training_test_results/unparametrized_circuit_loss.txt'
        the_filename2 = '../data/training_test_results/unparametrized_circuit_validation_loss.txt'
    elif the_choice == 'parametrized':
        the_filename1 = '../data/training_test_results/parametrized_circuit_loss.txt'
        the_filename2 = '../data/training_test_results/parametrized_circuit_validation_loss.txt'

    # saving the loss value for each epoch
    np.savetxt(the_filename1, epoch_loss)
    np.savetxt(the_filename2, validation_loss)

    # plotting the loss value for each epoch
    plt.plot(range(the_n_epochs), epoch_loss, label='training')
    plt.plot(range(the_n_epochs), validation_loss, label='validation')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss per Epoch')
    plt.legend(loc="upper right")
    plt.show()

    return the_weights


"""
here the_validation_set must be a list tuple (graph, output)!!!!!
"""


def validation_qgnn(the_validation_loader, the_weights, the_choice: str = 'parametrized', the_n_layers=3):
    """
    :param the_validation_loader: DataLoader object of the validation set
    :param the_weights: parameters to insert in the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param the_n_layers: numbers of layers of the quantum circuit
    :return: the_validation_loss: list of loss values for each point of the validation set after prediction
    """

    the_validation_set = []

    for _, item in enumerate(the_validation_loader):
        val = (to_networkx(data=item[0][0], node_attrs=['state', 'p_norm', 'theta'],
                           edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), item[1][0])
        the_validation_set.append(val)

    # define a list of ground truth values
    the_validation_truth = [element[1].detach().numpy() for element in the_validation_set]
    # define a list of prediction
    the_validation_predictions = predict(the_validation_set, the_weights, the_n_layers, the_choice)
    # the_validation_truth = np.array(the_validation_truth, dtype=object)
    assert len(the_validation_truth) == len(
        the_validation_predictions), "The number of predictions and true labels is not equal"

    the_validation_loss = get_mse(the_validation_predictions, the_validation_truth)

    return the_validation_loss.detach().numpy()


"""
function for predicting and plotting the test_set
"""


def test_prediction(the_test_loader, the_params, the_n_layers=3, the_choice: str = 'parametrized'):
    """
    this function compute the predicted outputs of unknonw datas (testset) and compare them
    to true output of them with a plot
    :param: the_test_loader: DataLoader object of the test set
    :param: the_params: parameters to insert in the quantum circuit
    :param: the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: None
    """

    targets = []

    # here I take each element in the_test_loader and reconvert it as a nextowrkx graph object
    for _, item in enumerate(the_test_loader):
        pred = (to_networkx(data=item[0][0], node_attrs=['state', 'p_norm', 'theta'],
                            edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), item[1][0])
        targets.append(pred)

    # convert each element from torch tensor into numpy array for the plot
    truth = [i[1].detach().numpy() for i in targets]  # here I define a list of the true values
    angles = [i[0].nodes[0]['theta'] for i in targets]  # here I build a list of scattering angles values
    targets = predict(targets, the_params, the_n_layers, the_choice)
    targets = [i.detach().numpy() for i in targets]  # here I build a list of predicted outputs

    # plotting lines
    plt.plot(angles, targets, 'ro', label='predictions')
    plt.plot(angles, truth, 'bs', label='ground truth')
    plt.xlabel('Scatteting Angle (rad)')
    plt.ylabel('Squared Matrix Element')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    np.savetxt('../data/training_test_results/test_outcomes.txt', targets)
