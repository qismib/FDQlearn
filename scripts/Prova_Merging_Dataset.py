from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, standardization
from qgraph import total_test_prediction, merged_train_qgnn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


def main(n_layers, max_epoch, the_file: str, the_train_file: str, the_val_file: str,
         the_param_file: str, choice: str, batch_size: int = 20, the_elem: int = 500, fold: int = 5, massive: bool = False):
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

    # fixing the seed
    random.seed(0)

    # I put kinetic_params = 1 if we're in the massless regime
    # I put kinetic_params = 2 if we're in the massive regime
    if massive is False:
        kinetic_num = 1
    else:
        kinetic_num = 2

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
        total_num_params = n_layers * (m + n + kinetic_num) + obs_params # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

    elif choice == 'parametrized':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        obs_params = 2  # number of parameters of the observable
        total_num_params = n_layers * (m + n + kinetic_num) + l + obs_params  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

    elif choice == 'fully_parametrized':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        obs_params = 2  # number of parameters of the observable
        total_num_params = n_layers * (m + n + kinetic_num) + 3 * l + obs_params # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

    elif choice == 'fully_connected':
        l = len(q_dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        obs_params = 2  # number of parameters of the observable
        total_num_params = n_layers * (n * (n - 1) // 2 + n + kinetic_num) + l + obs_params # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

    # Splitting q_dataset into training and validation set
    cross_set, test_set = train_test_split(q_dataset, train_size=0.8)

    dim_fold = len(cross_set) // fold

    train_loss = [0] * fold
    val_s_loss = [0] * fold
    val_t_loss = [0] * fold
    final_params = [0] * fold
    train_set = []

    # doing a k-fold cross validation
    for i in range(fold):
        # Splitting q_dataset into training and validation set
        validation_set = cross_set[i * dim_fold:(i + 1) * dim_fold]
        training_set = []
        for elem in cross_set:
            if elem not in validation_set:
                training_set.append(elem)

        train_set.append(training_set)

        init_params = 0.01 * torch.randn(total_num_params, dtype=torch.float)
        init_params.requires_grad = True
        print(init_params)

        validation_s_set = []
        validation_t_set = []
        for v_elem in validation_set:
            if v_elem[0]['scattering'] == 'e_e_s':
                validation_s_set.append(v_elem)
            elif v_elem[0]['scattering'] == 'e_e_t':
                validation_t_set.append(v_elem)

        # Building DataLoaders for each set
        training_loader = DataLoader(training_set, batch_size=batch_size)
        validation_loader = DataLoader(validation_set)
        validation_s_loader = DataLoader(validation_s_set)
        validation_t_loader = DataLoader(validation_t_set)

        final_params[i], train_loss[i], val_s_loss[i], val_t_loss[i] = merged_train_qgnn(training_loader, validation_s_loader,
                                                                                         validation_t_loader, init_params, max_epoch,
                                                                                         n_layers, choice, massive=massive)

        print("finished the ", (i + 1), " cross validation process")
        print('---------------------------------------------------------------------')

    cross_train_loss = []
    cross_train_loss_std = []
    for j in range(len(train_loss[0])):
        mean = 0
        mean2 = 0
        for i in range(fold):
            mean = mean + train_loss[i][j] / fold
            mean2 = mean2 + train_loss[i][j] * train_loss[i][j] / fold

        cross_train_loss.append(mean)
        cross_train_loss_std.append(np.sqrt(mean2 - mean * mean))

    cross_train_loss = np.array(cross_train_loss)
    cross_train_loss_std = np.array(cross_train_loss_std)

    print('average loss after the training process:', cross_train_loss[-1] / batch_size, ' +- ',
          cross_train_loss_std[-1] / batch_size)

    cross_val_s_loss = []
    cross_val_s_loss_std = []
    for j in range(len(val_s_loss[0])):
        mean = 0
        mean2 = 0
        for i in range(fold):
            mean = mean + val_s_loss[i][j] / fold
            mean2 = mean2 + val_s_loss[i][j] * val_s_loss[i][j] / fold

        cross_val_s_loss.append(mean)
        cross_val_s_loss_std.append(np.sqrt(mean2 - mean * mean))

    cross_val_s_loss = np.array(cross_val_s_loss)
    cross_val_s_loss_std = np.array(cross_val_s_loss_std)

    print('average validation loss for s-channel at the end of the training:', cross_val_s_loss[-1], ' +- ', cross_val_s_loss_std[-1])

    cross_val_t_loss = []
    cross_val_t_loss_std = []
    for j in range(len(val_t_loss[0])):
        mean = 0
        mean2 = 0
        for i in range(fold):
            mean = mean + val_t_loss[i][j] / fold
            mean2 = mean2 + val_t_loss[i][j] * val_t_loss[i][j] / fold

        cross_val_t_loss.append(mean)
        cross_val_t_loss_std.append(np.sqrt(mean2 - mean * mean))

    cross_val_t_loss = np.array(cross_val_t_loss)
    cross_val_t_loss_std = np.array(cross_val_t_loss_std)

    print('average validation loss for t-channel at the end of the training:', cross_val_t_loss[-1], ' +- ', cross_val_t_loss_std[-1])

    # saving the loss value for each epoch
    np.savetxt(the_train_file, cross_train_loss)
    np.savetxt(the_val_file, cross_val_s_loss)
    np.savetxt(the_val_file, cross_val_t_loss)

    # plotting the loss value for each epoch
    plt.plot(range(max_epoch), cross_train_loss, color='#CC4F1B')
    plt.fill_between(range(max_epoch), cross_train_loss - cross_train_loss_std, cross_train_loss + cross_train_loss_std,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss per Epoch')
    plt.title('Loss per epoch - training set ')
    plt.show()
    plt.plot(range(max_epoch), cross_val_s_loss, color='#1B2ACC')
    plt.fill_between(range(max_epoch), cross_val_s_loss - cross_val_s_loss_std, cross_val_s_loss + cross_val_s_loss_std,
                     alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss per Epoch')
    plt.title('Loss per epoch - s-channel validation set ')
    plt.show()
    plt.plot(range(max_epoch), cross_val_t_loss, color='#00FF00')
    plt.fill_between(range(max_epoch), cross_val_t_loss - cross_val_t_loss_std, cross_val_t_loss + cross_val_t_loss_std,
                     alpha=0.5, edgecolor='#00FF00', facecolor='#99FF99')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss per Epoch')
    plt.title('Loss per epoch - t-channel validation set ')
    plt.show()

    init_params = 0.01 * torch.randn(total_num_params, dtype=torch.float)
    init_params.requires_grad = True
    cross_loader = DataLoader(cross_set, batch_size=batch_size)
    test_loader = DataLoader(test_set)
    test_params, _, _, _ = merged_train_qgnn(cross_loader, test_loader, init_params, max_epoch,
                                             n_layers, choice, massive=massive)
    total_test_prediction(test_loader, test_params, n_layers, choice, massive)


# fixing the seeds:
torch.manual_seed(12345)
np.random.seed(12345)


num_layers = 3
num_epoch = 50
kfold = 5
batch = 20
elements = 1500
massive_regime = False

file = '../data/dataset/QED_data_qed.csv'
train_file = '../data/training_test_results/merged_training/parametrized_total_train_loss.txt'
val_file = '../data/training_test_results/merged_training/parametrized_total_val_loss.txt'
test_pred_file = '../data/interference/parametrized_total_final_outcome.txt'
test_param_file = '../data/interference/parametrized_total_final_params.txt'

feature_map = 'fully_connected'  # Must be either "parametrized", "unparametrized", "fully_parametrized",
# or "fully_connected", it indicates the kind of feature map to use in training

main(num_layers, num_epoch, file, train_file, val_file, test_param_file, feature_map, batch, elements, kfold, massive=massive_regime)