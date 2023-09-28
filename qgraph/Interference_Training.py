import time
from pennylane import numpy as np
import torch
from torch import optim
from torch_geometric.utils import to_networkx
from qgraph import interference_circuit
import matplotlib.pyplot as plt


def interference_truth(theta):  # for massless regime
    q_e = np.sqrt(4*np.pi/137)
    return q_e**4*(1+np.cos(theta))**2/(2*(1-np.cos(theta)))


def interference_prediction(the_s_data, the_s_params, the_t_data, the_t_params, the_weights, the_layers, the_choice: str,
                            massive: bool = False):

    prediction = 2*interference_circuit(the_s_data, the_s_params, the_weights[0], the_t_data, the_t_params,
                                        the_weights[1], the_layers, the_choice, massive=massive)

    return prediction


def training_interference(s_loader, the_s_params, t_loader, the_t_params, the_init_weights, the_n_epochs: int = 100,
                          the_n_layers: list = [3,5],  the_choice: str = 'parametrized', massive: bool = False):
    """
    Training function used to tune the phases for the two Feynman diagrams
    :param  s_loader: DataLoader object of the s-channel training set
    :param the_s_params: final params for the s-channel QGNN
    :param  t_loader: DataLoader object of the t-channel training set
    :param the_t_params: final params for the s-channel QGNN
    :param the_init_weights: parameters to insert in the quantum circuit (only 2 values)
    :param  the_n_epochs: number of epochs of the training process
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: the_weights: list of the final weights after the training
    """

    the_weights = the_init_weights
    opt = optim.Adam([the_weights], lr=1e-3)  # initialization of the optimizer to use
    epoch_loss = []
    validation_loss = []
    s_set = []
    t_set = []

    for s_item, t_item in zip(s_loader, t_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = to_networkx(data=s_item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)
        t_element = to_networkx(data=t_item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)

        assert s_element.graph['theta'] == t_element.graph['theta'] and \
               s_element.graph['p_norm'] == t_element.graph['p_norm'], "the angles and the momenta must be the same"

        s_set.append(s_element)
        t_set.append(t_element)

    for epoch in range(the_n_epochs):
        costs = []
        starting_time = time.time()

        for s_data, t_data in zip(s_set, t_set):
            def opt_func():  # defining an optimization function for the training of the model

                assert s_data.graph['theta'] == t_data.graph['theta'] and \
                       s_data.graph['p_norm'] == t_data.graph['p_norm'], "the angles and the momenta must be the same"

                prediction = interference_prediction(s_data, the_s_params, t_data, the_t_params,
                                                     the_weights, the_n_layers, the_choice, massive=massive)

                truth = interference_truth(s_data.graph['theta'])

                loss = (prediction - torch.tensor(truth, dtype=torch.float))**2
                costs.append(loss.item())
                loss.backward()
                return loss

            opt.zero_grad()
            opt.step(opt_func)

        ending_time = time.time()
        elapsed = ending_time - starting_time

        training_loss = np.mean(costs)
        epoch_loss.append(training_loss)

        if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-10:
            the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            print('the training process stopped at epoch number ', epoch)
            break

        if epoch % 5 == 0:
            res = [epoch, training_loss, elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | Elapsed Time per Epoch: {:3f}".format(*res))

    # plotting the loss value for each epoch
    plt.plot(range(the_n_epochs), epoch_loss, label='training')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss per Epoch')
    plt.legend(loc="upper right")
    plt.show()

    return the_weights


def one_data_training(s_loader, the_s_params, t_loader, the_t_params, the_n_epochs: int = 100,
                      the_n_layers: list = [3,5],  the_choice: str = 'parametrized', massive: bool = False):
    """
    Training function used to tune the phases for the two Feynman diagrams
    :param  s_loader: DataLoader object of the s-channel training set
    :param the_s_params: final params for the s-channel QGNN
    :param  t_loader: DataLoader object of the t-channel training set
    :param the_t_params: final params for the s-channel QGNN
    :param  the_n_epochs: number of epochs of the training process
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: output: predictions of the circuit
    :return: truth: theoretical values of the interference term
    :return: angles: list of the angle's values for each data
    """

    s_set = []
    t_set = []
    inter_pred = []
    ground_truth = []
    angles = []

    for s_item, t_item in zip(s_loader, t_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = to_networkx(data=s_item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)
        t_element = to_networkx(data=t_item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)

        assert s_element.graph['theta'] == t_element.graph['theta'] and \
               s_element.graph['p_norm'] == t_element.graph['p_norm'], "the angles and the momenta must be the same"

        s_set.append(s_element)
        t_set.append(t_element)

        the_weights = 0.01 * torch.randn(2, dtype=torch.float)
        the_weights.requires_grad = True
        opt = optim.Adam([the_weights], lr=1e-3)  # initialization of the optimizer to use
        epoch_loss = []
        # validation_loss = []
        convergence = the_n_epochs
        training_loss = 0

        for epoch in range(the_n_epochs):
            costs = []
            starting_time = time.time()

            def opt_func():  # defining an optimization function for the training of the model

                assert s_element.graph['theta'] == t_element.graph['theta'] and \
                       s_element.graph['p_norm'] == t_element.graph['p_norm'], "the angles and the momenta must be the same"

                prediction = interference_prediction(s_element, the_s_params, t_element, the_t_params,
                                                       the_weights, the_n_layers, the_choice, massive=massive)

                truth = interference_truth(s_element.graph['theta'])

                loss = (prediction - torch.tensor(truth, dtype=torch.float))**2
                costs.append(loss.item())
                loss.backward()
                return loss

            opt.zero_grad()
            opt.step(opt_func)

            training_loss = np.mean(costs)
            epoch_loss.append(training_loss)

            if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-10:
                convergence = epoch + 1  # Have to add 1 for plotting the right number of epochs
                break

        ending_time = time.time()
        elapsed = ending_time - starting_time

        if len(s_set) % 25 == 0:
            res = [len(s_set), convergence, training_loss, elapsed]
            print("Element: {:2d} | Epoch: {:2d} | Training loss: {:3f} | Elapsed Time per Epoch: {:3f}".format(*res))

        inter_pred.append(interference_prediction(s_element, the_s_params, t_element, the_t_params, the_weights,
                                                  the_n_layers, the_choice, massive=massive))
        ground_truth.append(interference_truth(s_element.graph['theta']))
        angles.append(s_element.graph['theta'])

    return inter_pred, ground_truth, angles


def interference_test(s_test_loader, the_s_params, t_test_loader, the_t_params, the_final_weights, the_n_layers: list = [3,5],
                      the_choice: str = 'parametrized', massive: bool = False):
    """
    Testing function used to check if the interference circuit works
    :param  s_test_loader: DataLoader object of the s-channel training set
    :param the_s_params: final params for the s-channel QGNN
    :param  t_test_loader: DataLoader object of the t-channel training set
    :param the_t_params: final params for the s-channel QGNN
    :param the_final_weights: parameters to insert in the quantum circuit after the tuning (only 2 values)
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: output: predictions of the circuit
    :return: truth: theoretical values of the interference term
    :return: angles: list of the angle's values for each data
    """

    assert len(s_test_loader) == len(t_test_loader), "the length of the Dataloaders must be the same"

    output = []
    angles = []
    truth = []

    for s, t in zip(s_test_loader, t_test_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = to_networkx(data=s[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)
        t_element = to_networkx(data=t[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)

        assert s_element.graph['theta'] == t_element.graph['theta'] and \
               s_element.graph['p_norm'] == t_element.graph['p_norm'], "the angles and the momenta must be the same"

        output.append(interference_prediction(s_element, the_s_params, t_element, the_t_params, the_final_weights,
                                              the_n_layers, the_choice, massive=massive))
        truth.append(interference_truth(s_element.graph['theta']))
        angles.append(s_element.graph['theta'])

    return output, truth, angles


def interference_gauge_setting(s_loader, the_s_params, t_loader, the_t_params, the_n_layers: list = [3,5],
                               the_choice: str = 'parametrized', massive: bool = False):
    """
    Hard computing function used to tune the phases for the two Feynman diagrams
    :param  s_loader: DataLoader object of the s-channel training set
    :param the_s_params: final params for the s-channel QGNN
    :param  t_loader: DataLoader object of the t-channel training set
    :param the_t_params: final params for the s-channel QGNN
    :param the_n_layers: numbers of layers of the quantum circuit
    :param  the_choice: kind of feature map to use in the quantum circuit (either 'parametrized' or 'unparametrized')
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: output: predictions of the circuit
    :return: truth: theoretical values of the interference term
    :return: angles: list of the angle's values for each data
    """

    gamma = np.linspace(0, 2*np.pi,75)
    delta = np.linspace(0, 2*np.pi,75)

    s_set = []
    t_set = []
    total_loss = []
    inter_pred = []
    ground_truth = []
    angles = []

    for s_item, t_item in zip(s_loader, t_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = to_networkx(data=s_item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)
        t_element = to_networkx(data=t_item[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                edge_attrs=['mass', 'spin', 'charge'], to_undirected=True)

        assert s_element.graph['theta'] == t_element.graph['theta'] and \
               s_element.graph['p_norm'] == t_element.graph['p_norm'], "the angles and the momenta must be the same"

        s_set.append(s_element)
        t_set.append(t_element)

        loss = []
        prediction = []

        is_looping = True
        starting_time = time.time()

        for i in gamma:
            for j in delta:

                assert s_element.graph['theta'] == t_element.graph['theta'] and \
                       s_element.graph['p_norm'] == t_element.graph['p_norm'], "the angles and the momenta must be the same"

                prediction.append(interference_prediction(s_element, the_s_params, t_element, the_t_params, [i, j],
                                                          the_n_layers, the_choice, massive=massive))

                truth = interference_truth(s_element.graph['theta'])

                loss.append((prediction[-1] - torch.tensor(truth, dtype=torch.float)) ** 2)

                if loss[-1] < 1e-8:
                    is_looping = False
                    break

            if not is_looping:
                break

        loss = torch.tensor([loss])
        total_loss.append(torch.min(loss))
        min_index = torch.argmin(loss)
        inter_pred.append(prediction[min_index])
        ground_truth.append(interference_truth(s_element.graph['theta']))
        angles.append(s_element.graph['theta'])

        ending_time = time.time()
        elapsed = ending_time - starting_time

        if len(s_set) % 5 == 0:
            res = [len(s_set)-1, elapsed]
            print("Element: {:2d} | Elapsed Time per Data: {:3f}".format(*res))

    return inter_pred, ground_truth, angles, total_loss