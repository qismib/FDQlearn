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
    return sum([(predictions[i] - torch.tensor(ground_truth[i], dtype=torch.float)) ** 2 for i in range(n)])


"""
here the_training_set must be a list tuple (graph, output)!!!!!
"""


def train_qgnn(the_training_loader, the_validation_loader, the_init_weights, the_n_epochs, the_train_file: str,
               the_val_file: str, the_n_layers=3, the_choice: str = 'parametrized'):
    """
    Version of training function for a dataset composed by a single Feynman diagram
    :param  the_training_loader: DataLoader object of the training set
    :param the_validation_loader: DataLoader object of the validation set
    :param the_init_weights: parameters to insert in the quantum circuit
    :param  the_n_epochs: number of epochs of the training process
    :param the_train_file: file where I save the training loss function per epoch
    :param the_val_file: file where I save the validation loss function per epoch
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: the_weights: list of the final weights after the training
    """

    the_weights = the_init_weights
    opt = optim.Adam([the_weights], lr=1e-3)  # initialization of the optimizer to use
    epoch_loss = []
    validation_loss = []

    for epoch in range(the_n_epochs):

        costs = []
        starting_time = time.time()

        for _, item in enumerate(the_training_loader):
            mini_batch = []
            # converting for each batch any DataLoader item into a list of tuples of networkx graph
            # object and the corresponding output
            mini_batch = [(to_networkx(data=item[0][i], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                       edge_attrs=['mass', 'spin', 'charge'], to_undirected=True),
                           item[1][i]) for i in range(len(item[0]))]

            def opt_func():  # defining an optimization function for the training of the model
                mini_batch_predictions = predict(mini_batch, the_weights, the_n_layers, the_choice)
                mini_batch_truth = [element[1] for element in mini_batch]
                loss = get_mse(mini_batch_predictions, mini_batch_truth)
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

        if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-5:
            the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            break

        if epoch % 5 == 0:
            res = [epoch, training_loss, the_val[0], elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | Validation loss: {:3f} | Elapsed Time per Epoch: {:3f}".format(*res))

    # saving the loss value for each epoch
    np.savetxt(the_train_file, epoch_loss)
    np.savetxt(the_val_file, validation_loss)

    return the_weights


def merged_train_qgnn(the_training_loader, the_validation_s_loader, the_validation_t_loader, the_init_weights, the_n_epochs, the_train_file: str,
               the_val_file: str, the_n_layers=3, the_choice: str = 'parametrized'):
    """
    Version of training function for a complete dataset (with more than 1 Feynman diagram), for
    which I want to divide the loss of each diagram
    :param  the_training_loader: DataLoader object of the training set
    :param the_validation_s_loader: DataLoader object of the validation set of the Bhabha s-channel
    :param the_validation_t_loader: DataLoader object of the validation set of the Bhabha t-channel
    :param the_init_weights: parameters to insert in the quantum circuit
    :param  the_n_epochs: number of epochs of the training process
    :param the_train_file: file where I save the training loss function per epoch
    :param the_val_file: file where I save the validation loss function per epoch
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: the_weights: list of the final weights after the training
    """

    the_weights = the_init_weights
    opt = optim.Adam([the_weights], lr=1e-2)  # initialization of the optimizer to use
    epoch_loss = []
    validation_s_loss = []
    validation_t_loss = []

    for epoch in range(the_n_epochs):

        costs = []
        starting_time = time.time()

        for _, item in enumerate(the_training_loader):
            mini_batch = []
            # converting for each batch any DataLoader item into a list of tuples of networkx graph
            # object and the corresponding output
            mini_batch = [(to_networkx(data=item[0][i], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                       edge_attrs=['mass', 'spin', 'charge'], to_undirected=True),
                           item[1][i]) for i in range(len(item[0]))]

            def opt_func():  # defining an optimization function for the training of the model
                mini_batch_predictions = predict(mini_batch, the_weights, the_n_layers, the_choice)
                mini_batch_truth = [element[1] for element in mini_batch]
                loss = get_mse(mini_batch_predictions, mini_batch_truth)
                costs.append(loss.item())
                loss.backward()
                return loss

            opt.zero_grad()
            opt.step(opt_func)

        ending_time = time.time()
        elapsed = ending_time - starting_time

        training_loss = np.mean(costs)
        epoch_loss.append(training_loss)

        the_s_val = validation_qgnn(the_validation_s_loader, the_weights, the_choice, the_n_layers)
        the_t_val = validation_qgnn(the_validation_t_loader, the_weights, the_choice, the_n_layers)
        validation_s_loss.append(the_s_val)
        validation_t_loss.append(the_t_val)

        if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-5:
            the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            break

        if epoch % 5 == 0:
            res = [epoch, training_loss, the_s_val[0], the_t_val[0], elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | s-channel loss: {:3f} | t-channel loss: {:3f} | "
                  "Elapsed Time per Epoch: {:3f}".format(*res))

    validation_loss = np.concatenate((validation_s_loss, validation_t_loss))
    # saving the loss value for each epoch
    np.savetxt(the_train_file, epoch_loss)
    np.savetxt(the_val_file, validation_loss)

    # plotting the loss value for each epoch
    plt.plot(range(the_n_epochs), epoch_loss, label='training')
    plt.plot(range(the_n_epochs), validation_s_loss, label='validation s-channel')
    plt.plot(range(the_n_epochs), validation_t_loss, label='validation t-channel')
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
    Function for validation step (we do it after each epoch of the training process)
    :param the_validation_loader: DataLoader object of the validation set
    :param the_weights: parameters to insert in the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param the_n_layers: numbers of layers of the quantum circuit
    :return: the_validation_loss: list of loss values for each point of the validation set after prediction
    """

    the_validation_set = []

    for _, item in enumerate(the_validation_loader):
        val = (to_networkx(data=item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                           edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), item[1][0])
        the_validation_set.append(val)

    # define a list of ground truth values
    the_validation_truth = [element[1].detach().numpy() for element in the_validation_set]
    # define a list of prediction
    the_validation_predictions = predict(the_validation_set, the_weights, the_n_layers, the_choice)
    # the_validation_truth = np.array(the_validation_truth, dtype=object)
    assert len(the_validation_truth) == len(the_validation_predictions), "The number of predictions and true labels is not equal"

    the_validation_loss = get_mse(the_validation_predictions, the_validation_truth)

    return the_validation_loss.detach().numpy()


"""
function for predicting and plotting the test_set
"""


def test_prediction(the_test_loader, the_params, the_test_file: str, the_truth_file: str, the_n_layers=3, the_choice: str = 'parametrized'):
    """
    this function compute the predicted outputs of unknonw datas (testset) and compare them
    to true output of them with a plot
    :param: the_test_loader: DataLoader object of the test set
    :param: the_params: parameters to insert in the quantum circuit
    :param: the_test_file: file where I save the predictions of the test set
    :param: the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: None
    """

    targets = []

    # here I take each element in the_test_loader and reconvert it as a nextowrkx graph object
    for _, item in enumerate(the_test_loader):

        with torch.no_grad():
            pred = (to_networkx(data=item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), item[1][0])
        targets.append(pred)

    # convert each element from torch tensor into numpy array for the plot
    truth = [i[1].detach().numpy() for i in targets]  # here I define a list of the true values
    angles = [i[0].graph['theta'] for i in targets]  # here I build a list of scattering angles values
    targets = predict(targets, the_params, the_n_layers, the_choice)
    targets = [i.detach().numpy() for i in targets]  # here I build a list of predicted outputs

    # plotting lines
    plt.plot(angles, targets, 'ro', label='predictions')
    plt.plot(angles, truth, 'bs', label='ground truth')
    plt.xlabel('Scattering Angle (rad)')
    plt.ylabel('Squared Matrix Element')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    np.savetxt(the_truth_file, truth)
    np.savetxt(the_test_file, targets)


def total_test_prediction(the_test_loader, the_params, the_y, the_n_layers=3, the_choice: str = 'parametrized'):
    """
    this function compute the predicted outputs of unknonw datas (testset) and compare them
    to true output of them with a plot
    :param: the_test_loader: DataLoader object of the test set
    :param: the_params: parameters to insert in the quantum circuit
    :param: the_y: list with the mean and the standard deviation of the output for the inverse transformation
    :param: the_n_layers: numbers of layers of the quantum circuit
    :param:  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: None
    """

    targets_e_mu_s = []
    targets_e_e_s = []
    targets_e_e_t = []

    # here I take each element in the_test_loader and reconvert it as a nextowrkx graph object
    for _, item in enumerate(the_test_loader):
        pred = (to_networkx(data=item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                            edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), item[1][0])
        # now I divide the kind of feynman diagrams to make predictions
        if pred[0].graph['scattering'] == 'e_mu_s':  # e+e- --> mu+mu- scattering
            targets_e_mu_s.append(pred)
        elif pred[0].graph['scattering'] == 'e_e_s':
            targets_e_e_s.append(pred)  # s-channel for Bhabha scattering
        elif pred[0].graph['scattering'] == 'e_e_t':
            targets_e_e_t.append(pred)  # t-channel for Bhabha scattering

    # convert each element from torch tensor into numpy array for the plot
    truth_e_mu_s = [(i[1]*the_y[1] + the_y[0]).detach().numpy() for i in targets_e_mu_s]  # here I define a list of the true values
    angles_e_mu_s = [i[0].graph['theta'] for i in targets_e_mu_s]  # here I build a list of scattering angles values
    targets_e_mu_s = predict(targets_e_mu_s, the_params, the_n_layers, the_choice)
    targets_e_mu_s = [(i*the_y[1] + the_y[0]).detach().numpy() for i in targets_e_mu_s]  # here I build a list of predicted outputs

    # plotting lines
    plt.figure(1)
    plt.plot(angles_e_mu_s, targets_e_mu_s, 'ro', label='predictions')
    plt.plot(angles_e_mu_s, truth_e_mu_s, 'bs', label='ground truth')
    plt.xlabel('Scattering Angle (rad)')
    plt.ylabel('Squared Matrix Element')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    np.savetxt('../data/training_test_results/test_outcomes_e_mu_s.txt', targets_e_mu_s)

    # convert each element from torch tensor into numpy array for the plot
    truth_e_e_s = [(i[1]*the_y[1] + the_y[0]).detach().numpy() for i in targets_e_e_s]  # here I define a list of the true values
    angles_e_e_s = [i[0].graph['theta'] for i in targets_e_e_s]  # here I build a list of scattering angles values
    targets_e_e_s = predict(targets_e_e_s, the_params, the_n_layers, the_choice)
    targets_e_e_s = [(i*the_y[1] + the_y[0]).detach().numpy() for i in targets_e_e_s]  # here I build a list of predicted outputs

    # plotting lines
    plt.figure(2)
    plt.plot(angles_e_e_s, targets_e_e_s, 'ro', label='predictions')
    plt.plot(angles_e_e_s, truth_e_e_s, 'bs', label='ground truth')
    plt.xlabel('Scattering Angle (rad)')
    plt.ylabel('Squared Matrix Element')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    np.savetxt('../data/training_test_results/test_outcomes_e_e_s.txt', targets_e_e_s)

    # convert each element from torch tensor into numpy array for the plot
    truth_e_e_t = [(i[1]*the_y[1] + the_y[0]).detach().numpy() for i in targets_e_e_t]  # here I define a list of the true values
    angles_e_e_t = [i[0].graph['theta'] for i in targets_e_e_t]  # here I build a list of scattering angles values
    targets_e_e_t = predict(targets_e_e_t, the_params, the_n_layers, the_choice)

    targets_e_e_t = [(i*the_y[1] + the_y[0]).detach().numpy() for i in targets_e_e_t]  # here I build a list of predicted outputs

    # plotting lines
    plt.figure(3)
    plt.plot(angles_e_e_t, targets_e_e_t, 'ro', label='predictions')
    plt.plot(angles_e_e_t, truth_e_e_t, 'bs', label='ground truth')
    plt.xlabel('Scattering Angle (rad)')
    plt.ylabel('Squared Matrix Element')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    np.savetxt('../data/training_test_results/test_outcomes_e_e_t.txt', targets_e_e_t)


"""
trial version to check the behaviour of the  parameters associated to the ZZ-layers
"""


def check_train(the_training_loader, the_validation_s_loader, the_validation_t_loader, the_init_weights, the_n_epochs,
                the_l: int, the_m: int, the_train_file: str, the_val_file: str, the_n_layers=3,
                the_choice: str = 'parametrized'):
    """
    :param  the_training_loader: DataLoader object of the training set
    :param the_validation_s_loader: DataLoader object of the validation set of the Bhabha s-channel
    :param the_validation_t_loader: DataLoader object of the validation set of the Bhabha t-channel
    :param the_init_weights: parameters to insert in the quantum circuit
    :param  the_n_epochs: number of epochs of the training process
    :param  the_l: number of parameters in the feature map
    :param  the_m: number of  ZZ layers in the ansatz
    :param the_train_file: file where I save the training loss function per epoch
    :param the_val_file: file where I save the validation loss function per epoch
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :return: the_weights: list of the final weights after the training
    """

    the_weights = the_init_weights
    opt = optim.Adam([the_weights], lr=1e-2)  # initialization of the optimizer to use
    epoch_loss = []
    validation_s_loss = []
    validation_t_loss = []
    edge_params = [0.]*the_m
    single_param = []
    for i in range(the_m):
        edge_params[i] = [the_weights[the_l+i].detach().numpy()]

    single_param.append(the_weights[the_l+the_m].detach().numpy())

    assert len(edge_params) == the_m, 'non è corretto'

    for epoch in range(the_n_epochs):

        costs = []
        starting_time = time.time()

        for _, item in enumerate(the_training_loader):
            mini_batch = []
            # converting for each batch any DataLoader item into a list of tuples of networkx graph
            # object and the corresponding output
            mini_batch = [(to_networkx(data=item[0][i], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                       edge_attrs=['mass', 'spin', 'charge'], to_undirected=True),
                           item[1][i]) for i in range(len(item[0]))]

            def opt_func():  # defining an optimization function for the training of the model
                mini_batch_predictions = predict(mini_batch, the_weights, the_n_layers, the_choice)
                mini_batch_truth = [element[1] for element in mini_batch]
                loss = get_mse(mini_batch_predictions, mini_batch_truth)
                costs.append(loss.item())
                loss.backward()
                return loss

            opt.zero_grad()
            opt.step(opt_func)

        for i in range(the_m):
            edge_params[i].append(the_weights[the_l + i].detach().numpy())

        single_param.append(the_weights[the_l+the_m].detach().numpy())

        ending_time = time.time()
        elapsed = ending_time - starting_time

        training_loss = np.mean(costs)
        epoch_loss.append(training_loss)

        the_s_val = validation_qgnn(the_validation_s_loader, the_weights, the_choice, the_n_layers)
        the_t_val = validation_qgnn(the_validation_t_loader, the_weights, the_choice, the_n_layers)
        validation_s_loss.append(the_s_val)
        validation_t_loss.append(the_t_val)

        if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-5:
            the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            break

        if epoch % 5 == 0:
            res = [epoch, training_loss, the_s_val[0], the_t_val[0], elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | s-channel loss: {:3f} | t-channel loss: {:3f} | "
                  "Elapsed Time per Epoch: {:3f}".format(*res))

    validation_loss = np.concatenate((validation_s_loss, validation_t_loss))
    # saving the loss value for each epoch
    np.savetxt(the_train_file, epoch_loss)
    np.savetxt(the_val_file, validation_loss)

    # plotting the loss value for each epoch
    plt.plot(range(the_n_epochs), epoch_loss, label='training')
    plt.plot(range(the_n_epochs), validation_s_loss, label='validation s-channel')
    plt.plot(range(the_n_epochs), validation_t_loss, label='validation t-channel')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss per Epoch')
    plt.legend(loc="upper right")
    plt.show()

    for i in range(len(edge_params)):
        plt.plot(range(len(edge_params[i])), edge_params[i], label=str(i)+'-th edge parameter evolution')
    plt.plot(range(len(single_param)), single_param, label='singleparameter of node feature encoding')
    plt.legend(loc='upper right')
    plt.show()

    plt.ylabel('Values of the edge parameters during training')
    plt.xlabel('number of updates')
    plt.legend(loc='upper right')
    plt.show()

    return the_weights

