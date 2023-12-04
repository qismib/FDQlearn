import os.path as osp
import ast
import torch
import pandas as pd
from pennylane import numpy as np
from sklearn.preprocessing import StandardScaler


def __get_node_features__(diagram):
    """
    function that returns the node features of a single data of the dataset
    :param: diagram: row of the csv file of the dataset
    :return: x: torch tensor of the node feature vectors (which are 3D)
    [Number of Nodes, 3]
    """
    x = ast.literal_eval(diagram.loc['x'])
    x = torch.tensor(x, dtype=torch.float, requires_grad=False).view(-1, 3)
    return x


def __get_edge_features__(diagram):
    """
    function that returns the edge features of a single data of the dataset
    :param: diagram: row of the csv file of the dataset
    :return: torhc tensor of the edge feature vectors (which are 12D)
    [Number of Edges, 11]
    """
    attr = ast.literal_eval(diagram.loc['edge_attr'])
    # for only QED I TAKE INTO ACCOUNT ONLY ELECTRIC CHARGE Q INSTEAD OF Y AND I3 (BOTH LEFT AND RIGHT)
    for i in attr:
        i[2] = i[2] + i[3] / 2
        del i[3:]
    return torch.tensor(attr, dtype=torch.float, requires_grad=False).view(-1, 3)


def __get_adj_list__(diagram):
    """
    function that returns the adjacency list of a single data of the dataset
    :param: diagram: row of the csv file of the dataset
    :return: torch tensor of the adjacency vectors (which are 2D)
    [2, Number of Edges]
    """
    adj_list = ast.literal_eval(diagram.loc['edge_index'])
    for i in range(len(adj_list[0])):
        adj_list[0][i] -= 1
        adj_list[1][i] -= 1
    x = torch.tensor(adj_list, dtype=torch.long, requires_grad=False).view(2, -1)
    return torch.transpose(x, 0, 1)


def __get_targets__(diagram):
    """
    function that returns the target value of a single data of the dataset
    :param: diagram: row of the csv file of the dataset
    :return:  torch tensor of the target y
    """
    y = diagram.loc['y']
    return torch.tensor(y, dtype=torch.float, requires_grad=False)


def __get_momentum__(diagram):
    """
    function that returns the momentum value of a single data of the dataset
    :param: diagram: row of the csv file of the dataset
    :return:  torch tensor of the momentum p
    """
    p = diagram.loc['p']
    return torch.tensor(p, dtype=torch.float, requires_grad=False)


def __get_scattering_angle__(diagram):
    """
    function that returns the scattering angle value of a single data of the dataset
    :param: diagram: row of the csv file of the dataset
    :return:  torch tensor of the angle theta
    """
    theta = diagram.loc['theta']
    return torch.tensor(theta, dtype=torch.float, requires_grad=False)


def __get_graph_index__(diagram):
    """
    function that returns as a string a label of the specific Feynman diagram
    :param: diagram: row of the csv file of the dataset
    :return:  torch tensor of the angle theta
    """
    index = diagram.loc['graph_index']
    return index


def FeynmanList(the_filename_path):
    """
    function FeynmanList takes as argument the filename of the csv dataset, it extract all the information in the csv
    per single data and put it into the list the_single_data. The function then returns the entire dataset as a list
    of list (the_final_dataset)
    :param: filename: string with the name of the csv file used as dataset
    :return: the_final_dataset: our dataset, a list of single datas
    """

    assert osp.isfile(the_filename_path) == True, "The file doesn't exist, maybe the path or the name of the file are incorrect"
    the_full_data = pd.read_csv(the_filename_path)
    the_final_dataset = []

    for i, feyndiag in the_full_data.iterrows():
        # node features
        the_x = __get_node_features__(feyndiag)
        # edge features
        the_edge_attr = __get_edge_features__(feyndiag)
        # adjacency list
        the_edge_index = __get_adj_list__(feyndiag)
        # targets
        the_y = __get_targets__(feyndiag)
        # get the momentum
        the_p = __get_momentum__(feyndiag)
        # get scattering angle
        the_theta = __get_scattering_angle__(feyndiag)
        # get the graph index
        the_index = __get_graph_index__(feyndiag)

        # create data list
        the_single_data = [the_x, the_edge_index, the_edge_attr, the_y, the_p, the_theta,
                           the_index]

        the_final_dataset.append(the_single_data)

    return the_final_dataset
