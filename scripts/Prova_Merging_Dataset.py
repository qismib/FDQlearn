from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, standardization
from qgraph import total_test_prediction, merged_train_qgnn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main(n_layers, max_epoch, the_file: str, the_train_file: str, the_val_file: str,
         the_param_file: str, choice: str, batch_size: int = 20, the_elem: int = 500, massive: bool = False):
    """
    Main function for QML algorithm for a dataset composed by different Feynman diagrams (s and t channel
    of Bhabha scattering)
    :param: n_layers: number of layers used for the ansatz
    :param: the_file: string with the path of the csv file from which I want to build the dataset
    :param: the_train_file: file for the training loss over the training process
    :param: the_val_file:  file for the validation loss at each epoch
    :param: the_param_file: file for the final values of the parameters of the network
    :param: the_choice: if we use a parametrized feature map or not
    :param: the_batch_size: batch size of the training dataset
    :param: the_elem: number of elements to pick randomly from the csv (number of elements of the dataset)
    :param: massive: boolean value that indicates whether we're in massive or massless regime
    :return: None
    """

    q_dataset = FeynmanDiagramDataset(the_file_path=the_file, the_n_elements=the_elem)

    # Splitting q_dataset into training and validation set
    training_set, validation_set = train_test_split(q_dataset, train_size=0.8)

    # standardization of the feature 'p_norm' and of the output 'y'
    y_stat, p_stat = standardization(training_set, validation_set, 0.5)

    validation_s_set = []
    validation_t_set = []
    for i in validation_set:
        if i[0]['scattering'] == 'e_e_s':
            validation_s_set.append(i)
        elif i[0]['scattering'] == 'e_e_t':
            validation_t_set.append(i)

    # Building DataLoaders for each set
    training_loader = DataLoader(training_set, batch_size=batch_size)
    # test_loader = DataLoader(test_set)
    validation_loader = DataLoader(validation_set)
    validation_s_loader = DataLoader(validation_s_set)
    validation_t_loader = DataLoader(validation_t_set)

    # I put kinetic_params = 1 if we're in the massless regime
    # I put kinetic_params = 2 if we're in the massive regime
    if massive == False:
        kinetic_num = 1
    else:
        kinetic_num = 2

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
    obs_params = 2

    if choice == 'unparametrized':
        init_params = 0.01 * torch.randn(n_layers * (m + n + kinetic_num) + obs_params, dtype=torch.float)  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        init_params.requires_grad = True
    elif choice == 'parametrized':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        init_params = 0.01 * torch.randn(n_layers * (m + n + kinetic_num) + l + obs_params, dtype=torch.float)  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        init_params.requires_grad = True
    elif choice == 'fully_parametrized':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        init_params = 0.01 * torch.randn(n_layers * (m + n + kinetic_num) + 3*l + obs_params, dtype=torch.float)  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        init_params.requires_grad = True
    elif choice == 'fully_connected':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        init_params = 0.01 * torch.randn(n_layers * (n*(n-1)//2 + n + kinetic_num) + l + obs_params, dtype=torch.float)  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME
        init_params.requires_grad = True
    else:
        print('choice can be either unparametrized, parametrized, fully_parametrized or fully_connected')

    final_params = merged_train_qgnn(training_loader, validation_s_loader, validation_t_loader, init_params, max_epoch,
                                     the_train_file, the_val_file, n_layers, choice, massive=massive)
    array_params = [i.detach().numpy() for i in final_params]
    np.savetxt(the_param_file, array_params)
    total_test_prediction(validation_loader, final_params, y_stat, n_layers, choice, massive=massive)


# fixing the seeds:
torch.manual_seed(12345)
np.random.seed(12345)


num_layers = 3
num_epoch = 30
batch = 20
elements = 750
massive_regime = False

file = '../data/dataset/QED_data_qed.csv'
train_file = '../data/training_test_results/merged_trianing/parametrized_total_train_loss.txt'
val_file = '../data/training_test_results/merged_trianing/parametrized_total_val_loss.txt'
test_pred_file = '../data/interference/parametrized_total_final_outcome.txt'
test_param_file = '../data/interference/parametrized_total_final_params.txt'

feature_map = 'fully_connected'  # Must be either "parametrized", "unparametrized", "fully_parametrized",
# or "fully_connected", it indicates the kind of feature map to use in training

main(num_layers, num_epoch, file, train_file, val_file, test_param_file, feature_map, batch, elements, massive=massive_regime)