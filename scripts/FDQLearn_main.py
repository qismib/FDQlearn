from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, standardization
from qgraph import train_qgnn, test_prediction
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


def main(n_layers, max_epoch, the_dataset_file: str, the_train_file: str, the_val_file: str,
         the_test_file: str, the_truth_file: str, the_param_file: str, the_standardization_file: str, choice: str,
         batch_size=20, the_elem=500, the_bandwidth=0.5):

    q_dataset = FeynmanDiagramDataset(the_file_path=the_dataset_file, the_n_elements=the_elem)

    # Splitting q_dataset into training and validation set
    training_set, validation_set = train_test_split(q_dataset, train_size=0.8)

    # y_stat, p_stat = standardization(training_set, validation_set)

    # Building DataLoaders for each set
    training_loader = DataLoader(training_set, batch_size=batch_size)
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
        obs_params = 2  # number of parameters of the observable
        init_params = 0.01 * torch.randn(n_layers * (m + n + 1) + obs_params, dtype=torch.float)  # IF YOU ADD THE MOMENTUM P YOU HAVE TO PUT 2 INSTEAD OF 1
        init_params.requires_grad = True
    elif choice == 'parametrized':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        obs_params = 2  # number of parameters of the observable
        init_params = 0.01 * torch.randn(n_layers * (m + n + 1) + l + obs_params, dtype=torch.float)  # IF YOU ADD THE MOMENTUM P YOU HAVE TO PUT 2 INSTEAD OF 1
        init_params.requires_grad = True
    elif choice == 'fully_parametrized':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        obs_params = 2  # number of parameters of the observable
        init_params = 0.01 * torch.randn(n_layers * (m + n + 1) + 3*l + obs_params, dtype=torch.float)  # IF YOU ADD THE MOMENTUM P YOU HAVE TO PUT 2 INSTEAD OF 1
        init_params.requires_grad = True

    print(init_params)

    final_params = train_qgnn(training_loader, validation_loader, init_params, max_epoch, the_train_file,
                              the_val_file, n_layers, choice)

    print(final_params)

    array_params = [i.detach().numpy() for i in final_params]
    np.savetxt(the_param_file, array_params)
    test_prediction(validation_loader, final_params, the_test_file, the_truth_file, n_layers, choice)

    # the_bandwidth = np.array([the_bandwidth])
    # np.savetxt(the_standardization_file, np.concatenate((y_stat, p_stat, the_bandwidth)))

    return final_params


# fixing the seeds:
torch.manual_seed(12345)
np.random.seed(12345)


num_layers = 5
num_epoch = 100
batch = 20
elements = 500
csv_file = '../data/dataset/QED_data_e_annih_e_t.csv'
train_file = '../data/training_test_results/parametrized_t_channel_train_loss.txt'
val_file = '../data/training_test_results/parametrized_t_channel_val_loss.txt'
test_pred_file = '../data/training_test_results/parametrized_t_channel_predictions.txt'
truth_file = '../data/training_test_results/parametrized_t_channel_ground_truth.txt'
final_params_file = '../data/interference/parametrized_channel_t_final_params.txt'
standardization_file = '../data/interference/parametrized_channel_t_standardization.txt'

feature_map = 'parametrized'  # Must be either "parametrized", "unparametrized" or "fully_connected", it indicates the
# kind of feature map to use in training

params = main(num_layers, num_epoch, csv_file, train_file, val_file, test_pred_file, truth_file, final_params_file,
              standardization_file, feature_map, batch, elements)