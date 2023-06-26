from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, standardization
from qgraph import train_qgnn, test_prediction
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


def main(n_layers, max_epoch, the_file: str, the_train_file: str, the_val_file: str,
         the_test_file: str, choice: str, batch_size=20, the_elem=500):

    q_dataset = FeynmanDiagramDataset(the_file_path=the_file, the_n_elements=the_elem)

    # Splitting q_dataset into training, test and validation set
    training_set, test_set = train_test_split(q_dataset, train_size=0.8)
    training_set, validation_set = train_test_split(training_set, train_size=0.8)

    y_stat, p_stat = standardization(training_set, validation_set, test_set, 0.5)

    # Building DataLoaders for each set
    training_loader = DataLoader(training_set, batch_size=batch_size)
    test_loader = DataLoader(test_set)
    validation_loader = DataLoader(validation_set)

    """
    CALCULATING THE NUMBER OF PARAMETERS I HAVE TO DEFINE FOR THE QGNN
    """
    init = 0
    fin = 0
    propagator = 0
    for i in q_dataset.dataset[0][0].nodes:  # here I use self.dataset (return a list of tuples:(nx.graph,output))
        #  attribute of FeynmanDiagramDataset class for counting the initial, final and propagator nodes
        if q_dataset.dataset[0][0].nodes[i]['state'][0] == 1:  # here I count initial-state nodes
            init += 1
        elif q_dataset.dataset[0][0].nodes[i]['state'][2] == 1:  # here I count final-state nodes
            fin += 1
        else:  # here I count propagator-state nodes
            propagator += 1
    m = int((init + fin + (propagator - 1) / 2) * propagator)  # number of combinations I can connect nodes
    n = len(q_dataset.dataset[0][0].nodes)  # total number of nodes

    if choice == 'unparametrized':
        init_params = 0.01 * torch.randn(n_layers * (m + n + 2), dtype=torch.float)
        init_params.requires_grad = True
    elif choice == 'parametrized':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        init_params = 0.01 * torch.randn(n_layers * (m + n + 2) + l, dtype=torch.float)
        init_params.requires_grad = True

    final_params = train_qgnn(training_loader, validation_loader, init_params, max_epoch, the_train_file,
                              the_val_file, n_layers, choice)
    array_params = [i.detach().numpy() for i in final_params]
    # np.savetxt('../data/training_test_results/'+choice + '_circuit_final_params.txt', array_params)
    np.savetxt(the_test_file, array_params)
    test_prediction(test_loader, final_params, n_layers, choice)

    return final_params


# fixing the seeds:
torch.manual_seed(12345)
np.random.seed(12345)


num_layers = 3
num_epoch = 30
batch = 20
elements = 500
file = '../data/dataset/QED_data_e_annih_e_s.csv'
train_file = 'loss_prova.txt'  # '../data/training_test_results/parametrized_s_channel_train_loss.txt'
val_file = 'val_prova.txt'  # '../data/training_test_results/parametrized_s_channel_val_loss.txt'
test_pred_file = 'pred_prova.txt'  # '../data/interference/parametrized_channel_s_final_params.txt'

feature_map = 'parametrized'  # Must be either "parametrized" or "unparametrized", it indicates the
# kind of feature map to use in training

params = main(num_layers, num_epoch, file, train_file, val_file, test_pred_file, feature_map, batch, elements)