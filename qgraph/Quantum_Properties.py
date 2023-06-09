from pennylane import numpy as np
from torch_geometric.utils import to_networkx
from qgraph import total_matrix_circuit, interference_circuit


def interference(the_s_loader, s_params, the_t_loader, t_params, the_n_layers,
                 the_choice: str = 'parametrized'):
    """
    function that computes the interference term for all the datas
    :param: the_s_loader: set of graphs representing all the s-channel diagrams
    :param: the_s_params: value of the final parameters for s-channel (after training)
    :param: the_t_channel: set of graphs representing all the t-channel diagrams
    :param: the_t_params: value of the final parameters for channel t (after training)
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :return: output: set of all the interference terms for all the points
    :return: angles: set of all the angles values of the dataset
    """
    assert len(the_s_loader) == len(the_t_loader), "the length of the Dataloaders must be the same"

    output = []
    angles = []

    for s, t in zip(the_s_loader, the_t_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = (to_networkx(data=s[0][0], node_attrs=['state', 'p_norm', 'theta'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), s[1][0])
        t_element = (to_networkx(data=t[0][0], node_attrs=['state', 'p_norm', 'theta'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), t[1][0])

        assert s_element[0].nodes[0]['theta'] == t_element[0].nodes[0]['theta'] and \
               s_element[0].nodes[0]['p_norm'] == t_element[0].nodes[0]['p_norm'], "the angles and the momenta must be the same"

        angles.append(s_element[0].nodes[0]['theta'])
        output.append(interference_circuit(s_element[0], s_params, t_element[0], t_params, the_n_layers, the_choice))

    np.savetxt('../data/interference/interference_outcomes.txt', output)

    return output, angles


def matrix_squared(the_s_loader, s_params, the_t_loader, t_params, the_n_layers,
                   the_choice: str = 'parametrized'):
    """
    function that computes the total matrix element for all the datas
    :param: the_s_loader: set of graphs representing all the s-channel diagrams
    :param: the_s_params: value of the final parameters for s-channel (after training)
    :param: the_t_channel: set of graphs representing all the t-channel diagrams
    :param: the_t_params: value of the final parameters for channel t (after training)
    :param: the_layers: number of layers
    :param: the_choice: string that tells which feature map we want to use
    :return: output: set of all the interference terms for all the points
    :return: angles: set of all the angles values of the dataset
    """

    assert len(the_s_loader) == len(the_t_loader), "the length of the Dataloaders must be the same"

    output = []
    angles = []
    for s, t in zip(the_s_loader, the_t_loader):
        # converting for each batch any DataLoader item into a list of tuples of networkx graph
        # object and the corresponding output
        s_element = (to_networkx(data=s[0][0], node_attrs=['state', 'p_norm', 'theta'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), s[1][0])
        t_element = (to_networkx(data=t[0][0], node_attrs=['state', 'p_norm', 'theta'],
                                 edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), t[1][0])
        assert s_element[0].nodes[0]['theta'] == t_element[0].nodes[0]['theta'] and \
               s_element[0].nodes[0]['p_norm'] == t_element[0].nodes[0]['p_norm'], "the angles and the momenta must be the same"

        angles.append(s_element[0].nodes[0]['theta'])
        output.append(total_matrix_circuit(s_element[0], s_params, t_element[0], t_params, the_n_layers, the_choice))

    np.savetxt('../data/interference/total_matrix_outcomes.txt', output)

    return output, angles