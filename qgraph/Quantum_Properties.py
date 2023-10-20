from pennylane import numpy as np
import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from qgraph import total_matrix_circuit, interference_circuit, predict, phase_estimation


def interference(the_s_loader, s_params, the_t_loader, t_params, the_n_layers, the_choice: str = 'parametrized',
                 massive: bool = False):
    """
    function that computes the interference term for all the datas
    :param: the_s_loader: set of graphs representing all the s-channel diagrams
    :param: s_params: value of the final parameters for s-channel (after training)
    :param: the_t_loader: set of graphs representing all the t-channel diagrams
    :param: z_params: value of the final parameters for channel t (after training)
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: output: set of all the interference terms for all the points
    :return: angles: set of all the angles values of the dataset
    :return: s_glob_phase: set of all the global phases for the s-channel
    :return: t_glob_phase: set of all the global phases for t-channel
    """
    assert len(the_s_loader) == len(the_t_loader), "the length of the Dataloaders must be the same"

    output = []
    angles = []
    s_truth = []
    t_truth = []
    s_set = []
    t_set = []
    s_glob_phase = []
    t_glob_phase = []

    for s, t in zip(the_s_loader, the_t_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = (to_networkx(data=s[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), s[1][0])
        t_element = (to_networkx(data=t[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), t[1][0])
        s_set.append(s_element)
        t_set.append(t_element)

        assert s_element[0].graph['theta'] == t_element[0].graph['theta'] and \
               s_element[0].graph['p_norm'] == t_element[0].graph['p_norm'], "the angles and the momenta must be the same"

        s_phase = global_phase(s_element, s_params, the_n_layers[0], the_choice, massive=massive)
        t_phase = global_phase(t_element, t_params, the_n_layers[1], the_choice, massive=massive)

        s_glob_phase.append(s_phase)
        t_glob_phase.append(t_phase)

        angles.append(s_element[0].graph['theta'])
        output.append(2*interference_circuit(s_element[0], s_params, s_phase, t_element[0], t_params, t_phase,
                                             the_n_layers, the_choice, massive=massive).detach().numpy())
        s_truth.append(s_element[1])
        t_truth.append(t_element[1])

    s_pred = predict(s_set, s_params, the_n_layers[0], the_choice, massive=massive)
    s_pred = [s.detach().numpy() for s in s_pred]
    t_pred = predict(t_set, t_params, the_n_layers[1], the_choice, massive=massive)
    t_pred = [t.detach().numpy() for t in t_pred]

    plt.plot(angles, s_truth, 'ro', label='s channel truth')
    plt.plot(angles, s_pred, 'bs', label='s channel pred')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(angles, t_truth, 'ro', label='t channel truth')
    plt.plot(angles, t_pred, 'bs', label='t channel pred')
    plt.legend(loc='upper right')
    plt.show()

    return output, angles, s_glob_phase, t_glob_phase


def matrix_squared(the_s_loader, s_params, the_t_loader, t_params, the_n_layers, the_choice: str = 'parametrized',
                   massive: bool = False):
    """
    function that computes the total matrix element for all the datas
    :param: the_s_loader: set of graphs representing all the s-channel diagrams
    :param: s_params: value of the final parameters for s-channel (after training)
    :param: the_t_loader: set of graphs representing all the t-channel diagrams
    :param: z_params: value of the final parameters for channel t (after training)
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: output: set of all the interference terms for all the points
    :return: angles: set of all the angles values of the dataset
    """

    assert len(the_s_loader) == len(the_t_loader), "the length of the Dataloaders must be the same"

    output = []
    angles = []

    for s, t in zip(the_s_loader, the_t_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = (to_networkx(data=s[0][0], graph_attrs=['scattering','p_norm', 'theta'], node_attrs=['state'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), s[1][0])
        t_element = (to_networkx(data=t[0][0], graph_attrs=['scattering','p_norm', 'theta'], node_attrs=['state'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), t[1][0])
        assert s_element[0].graph['theta'] == t_element[0].graph['theta'] and \
               s_element[0].graph['p_norm'] == t_element[0].graph['p_norm'], "the angles and the momenta must be the same"

        angles.append(s_element[0].graph['theta'])
        output.append(total_matrix_circuit(s_element[0], s_params, t_element[0], t_params, the_n_layers, the_choice, massive=massive))

    return output, angles


def global_phase(the_element, the_final_params, the_n_layers, the_choice: str = 'parametrized', massive: bool = False):
    """
    function that computes the global phase for all the datas
    :param: the_element: graph representing a Feynman diagram
    :param: the_final_params: value of the final parameters (after training)
    :param: the_n_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: glob_phase: global state of the state
    """

    state = phase_estimation(the_element[0], the_final_params, the_n_layers, the_choice, massive=massive)

    state = torch.angle(state[0])

    if torch.min(torch.abs(state)) == 0:
        glob_phase = 0
    else:
        glob_phase = torch.min(state).numpy()

    return glob_phase