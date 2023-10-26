import time
from pennylane import numpy as np
import torch
from torch import optim
from torch_geometric.utils import to_networkx
from qgraph import expect_value
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def predict(the_dataset, the_weights, the_n_layers, the_choice: str, massive: bool = False):
    """
    :param the_dataset: either validation or test set, list of datas
    :param the_weights: array of trained parameters
    :param the_n_layers: number of layers of the ansatz (depth of the circuit)
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param massive: boolean value that indicates whether we're in massive or massless regime
    :return: list of predictions
    """
    predictions = []
    the_circuit_weights = the_weights[:-2]
    the_observable_weights = the_weights[-2:]

    for element in the_dataset:
        probability = expect_value(element[0], the_n_layers, the_circuit_weights, the_choice, massive)
        output = torch.abs(the_observable_weights[0])*probability[0][0] + torch.abs(the_observable_weights[1])*probability[0][1]
        predictions.append(output)

    return predictions


def get_mse(predictions, ground_truth):
    """
    :param predictions: list of predictions of the QGNN
    :param  ground_truth: list of true labels
    :return: mse: sum of squared errors
    """
    n = len(predictions)
    assert len(ground_truth) == n, "The number of predictions and true labels is not equal"
    return sum([(predictions[i] - torch.tensor(ground_truth[i], dtype=torch.float))**2 for i in range(n)])


def relative_error(predictions, ground_truth):
    """
    :param predictions: list of predictions of the QGNN
    :param  ground_truth: list of true labels
    :return: rel_error: mean relative error
    """
    n = len(predictions)
    assert len(ground_truth) == n, "The number of predictions and true labels is not equal"
    error = np.array([np.abs(predictions[i] - ground_truth[i])/np.abs(ground_truth[i]) for i in range(n)])
    print(len(error))
    error = np.mean(error)
    return error


"""
here the_training_set must be a list tuple (graph, output)!!!!!
"""


def train_qgnn(the_training_loader, the_validation_loader, the_init_weights, the_n_epochs,
               the_n_layers=3, the_choice: str = 'parametrized', massive: bool = False):
    """
    Version of training function for a dataset composed by a single Feynman diagram
    :param  the_training_loader: DataLoader object of the training set
    :param the_validation_loader: DataLoader object of the validation set
    :param the_init_weights: parameters to insert in the quantum circuit
    :param  the_n_epochs: number of epochs of the training process
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param massive: boolean value that indicates whether we're in massive or massless regime
    :return: the_weights: list of the final weights after the training
    """

    the_weights = the_init_weights
    opt = optim.Adam([the_weights], lr=1e-2)  # initialization of the optimizer to use
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
                mini_batch_predictions = predict(mini_batch, the_weights, the_n_layers, the_choice, massive=massive)
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

        the_val = validation_qgnn(the_validation_loader, the_weights, the_choice, the_n_layers, massive=massive)
        validation_loss.append(the_val)

        # if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-10:
            # the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            # print('the training process stopped at epoch number ', epoch)
            # break

        if epoch % 5 == 0:
            res = [epoch, training_loss, the_val, elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | Validation loss: {:3f} | Elapsed Time per Epoch: {:3f}".format(*res))

    return the_weights, epoch_loss, validation_loss


def merged_train_qgnn(the_training_loader, the_validation_s_loader, the_validation_t_loader, the_init_weights,
                      the_n_epochs, the_n_layers=3, the_choice: str = 'parametrized', massive: bool = False):
    """
    Version of training function for a complete dataset (with more than 1 Feynman diagram), for
    which I want to divide the loss of each diagram
    :param  the_training_loader: DataLoader object of the training set
    :param the_validation_s_loader: DataLoader object of the validation set of the Bhabha s-channel
    :param the_validation_t_loader: DataLoader object of the validation set of the Bhabha t-channel
    :param the_init_weights: parameters to insert in the quantum circuit
    :param  the_n_epochs: number of epochs of the training process
    :param the_n_layers: numbers of layers of the quantum circuit
    :param the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param massive: boolean value that indicates whether we're in massive or massless regime
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
                mini_batch_predictions = predict(mini_batch, the_weights, the_n_layers, the_choice, massive=massive)
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

        the_s_val = validation_qgnn(the_validation_s_loader, the_weights, the_choice, the_n_layers, massive=massive)
        the_t_val = validation_qgnn(the_validation_t_loader, the_weights, the_choice, the_n_layers, massive=massive)
        validation_s_loss.append(the_s_val)
        validation_t_loss.append(the_t_val)

        if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-5:
            the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            print('the training process stopped at epoch number:', epoch)
            break

        if epoch % 5 == 0:
            res = [epoch, training_loss, the_s_val, the_t_val, elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | s-channel loss: {:3f} | t-channel loss: {:3f} | "
                  "Elapsed Time per Epoch: {:3f}".format(*res))

    # plotting the loss value for each epoch
    # plt.plot(range(the_n_epochs), epoch_loss, label='training')
    # plt.plot(range(the_n_epochs), validation_s_loss, label='validation s-channel')
    # plt.plot(range(the_n_epochs), validation_t_loss, label='validation t-channel')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Loss per Epoch')
    # plt.legend(loc="upper right")
    # plt.show()

    return the_weights, epoch_loss, validation_s_loss, validation_t_loss


"""
here the_validation_set must be a list tuple (graph, output)!!!!!
"""


def validation_qgnn(the_validation_loader, the_weights, the_choice: str = 'parametrized', the_n_layers=3, massive: bool = False):
    """
    Function for validation step (we do it after each epoch of the training process)
    :param the_validation_loader: DataLoader object of the validation set
    :param the_weights: parameters to insert in the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param the_n_layers: numbers of layers of the quantum circuit
    :param massive: boolean value that indicates whether we're in massive or massless regime
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
    the_validation_predictions = predict(the_validation_set, the_weights, the_n_layers, the_choice, massive=massive)
    # the_validation_truth = np.array(the_validation_truth, dtype=object)
    assert len(the_validation_truth) == len(the_validation_predictions), "The number of predictions and true labels is not equal"

    the_validation_loss = get_mse(the_validation_predictions, the_validation_truth)

    the_validation_loss = the_validation_loss.detach().numpy()
    return the_validation_loss


"""
function for predicting and plotting the test_set
"""


def test_prediction(the_test_loader, the_params, the_test_file: str, the_truth_file: str, the_angle_file: str,
                    the_momentum_file: str, the_n_layers=3, the_choice: str = 'parametrized', massive: bool = False):
    """
    this function compute the predicted outputs of unknown datas (testset) and compare them
    to true output of them with a plot
    :param: the_test_loader: DataLoader object of the test set
    :param: the_params: parameters to insert in the quantum circuit
    :param: the_test_file: file where I save the predictions of the test set
    :param: the_truth_file: file where I save the theoretical values of the test set
    :param: the_angle_file: file where I save the scattering angles of the test set
    :param: the_momentum_file: file where I save the momenta of the test set
    :param: the_n_layers: numbers of layers of the quantum circuit
    :param:  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param: massive: boolean value that indicates whether we're in massive or massless regime
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
    momentum = [i[0].graph['p_norm'] for i in targets]
    targets = predict(targets, the_params, the_n_layers, the_choice, massive=massive)
    targets = [i.detach().numpy() for i in targets]  # here I build a list of predicted outputs

    rel_error = relative_error(targets, truth)
    print('the relative error per element of the test set is:', rel_error)
    mse = get_mse(targets, truth)/len(targets)
    print('the mean squared error per element of the test set is:', mse)

    # plotting lines
    # plt.plot(angles, targets, 'ro', label='predictions')
    # plt.plot(angles, truth, 'bs', label='ground truth')
    # plt.title('|M|^2 prediction over the test set')
    # plt.xlabel('Scattering Angle (rad)')
    # plt.ylabel('Squared Matrix Element')
    # plt.legend(loc='upper right')
    # plt.grid(True)
    # plt.show()
    # if massive is True:
    #    ax = plt.axes(projection='3d')
    #    ax.scatter3D(angles, momentum, targets, color='red')
    #    ax.scatter3D(angles, momentum, truth, color='blue')
    #    ax.set_title('Squared Matrix Element')
    #    ax.set_xlabel('Scattering Angle (rad)', fontsize=12)
    #    ax.set_ylabel('Momentum of the particles', fontsize=12)
    #    ax.set_zlabel('Squared Matrix Element', fontsize=12)
    #    plt.show()

    np.savetxt(the_truth_file, truth)
    np.savetxt(the_test_file, targets)
    np.savetxt(the_angle_file, angles)
    if massive is True:
        np.savetxt(the_momentum_file, momentum)


def total_test_prediction(the_test_loader, the_params, the_n_layers=3, the_choice: str = 'parametrized', massive: bool = False):
    """
    this function compute the predicted outputs of unknonw datas (testset) and compare them
    to true output of them with a plot
    :param: the_test_loader: DataLoader object of the test set
    :param: the_params: parameters to insert in the quantum circuit
    :param: the_n_layers: numbers of layers of the quantum circuit
    :param:  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param: massive: boolean value that indicates whether we're in massive or massless regime
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
    truth_e_mu_s = [i[1].detach().numpy() for i in targets_e_mu_s]  # here I define a list of the true values
    angles_e_mu_s = [i[0].graph['theta'] for i in targets_e_mu_s]  # here I build a list of scattering angles values
    targets_e_mu_s = predict(targets_e_mu_s, the_params, the_n_layers, the_choice, massive=massive)
    targets_e_mu_s = [i.detach().numpy() for i in targets_e_mu_s]  # here I build a list of predicted outputs

    # plotting lines
    # plt.figure(1)
    # plt.plot(angles_e_mu_s, targets_e_mu_s, 'ro', label='predictions')
    # plt.plot(angles_e_mu_s, truth_e_mu_s, 'bs', label='ground truth')
    # plt.xlabel('Scattering Angle (rad)')
    # plt.ylabel('Squared Matrix Element')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.show()
    np.savetxt('../data/training_test_results/merged_training/test_truth_e_mu_s.txt', truth_e_mu_s)
    np.savetxt('../data/training_test_results/merged_training/angles_e_mu_s.txt', angles_e_mu_s)
    np.savetxt('../data/training_test_results/merged_training/test_outcomes_e_mu_s.txt', targets_e_mu_s)

    # convert each element from torch tensor into numpy array for the plot
    truth_e_e_s = [i[1].detach().numpy() for i in targets_e_e_s]  # here I define a list of the true values
    angles_e_e_s = [i[0].graph['theta'] for i in targets_e_e_s]  # here I build a list of scattering angles values
    targets_e_e_s = predict(targets_e_e_s, the_params, the_n_layers, the_choice, massive=massive)
    targets_e_e_s = [i.detach().numpy() for i in targets_e_e_s]  # here I build a list of predicted outputs

    # plotting lines
    # plt.figure(2)
    # plt.plot(angles_e_e_s, targets_e_e_s, 'ro', label='predictions')
    # plt.plot(angles_e_e_s, truth_e_e_s, 'bs', label='ground truth')
    # plt.xlabel('Scattering Angle (rad)')
    # plt.ylabel('Squared Matrix Element')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.show()
    np.savetxt('../data/training_test_results/merged_training/test_truth_e_e_s.txt', truth_e_e_s)
    np.savetxt('../data/training_test_results/merged_training/angles_e_e_s.txt', angles_e_e_s)
    np.savetxt('../data/training_test_results/merged_training/test_outcomes_e_e_s.txt', targets_e_e_s)

    # convert each element from torch tensor into numpy array for the plot
    truth_e_e_t = [i[1].detach().numpy() for i in targets_e_e_t]  # here I define a list of the true values
    angles_e_e_t = [i[0].graph['theta'] for i in targets_e_e_t]  # here I build a list of scattering angles values
    targets_e_e_t = predict(targets_e_e_t, the_params, the_n_layers, the_choice, massive=massive)

    targets_e_e_t = [i.detach().numpy() for i in targets_e_e_t]  # here I build a list of predicted outputs

    # plotting lines
    # plt.figure(3)
    # plt.plot(angles_e_e_t, targets_e_e_t, 'ro', label='predictions')
    # plt.plot(angles_e_e_t, truth_e_e_t, 'bs', label='ground truth')
    # plt.xlabel('Scattering Angle (rad)')
    # plt.ylabel('Squared Matrix Element')
    # plt.legend(loc='upper right')
    # plt.grid(True)
    # plt.show()
    np.savetxt('../data/training_test_results/merged_training/test_truth_e_e_t.txt', truth_e_e_t)
    np.savetxt('../data/training_test_results/merged_training/angles_e_e_t.txt', angles_e_e_t)
    np.savetxt('../data/training_test_results/merged_training/test_outcomes_e_e_t.txt', targets_e_e_t)