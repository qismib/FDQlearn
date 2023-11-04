from pennylane import numpy as np
import torch
from qgraph import train_qgnn, test_prediction, min_max, min_max_scaling
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


def model_evaluation(n_layers, max_epoch, dataset, the_train_file: str, the_train_std_file: str, the_val_file: str,
                     the_val_std_file: str, the_test_file: str, the_truth_file: str, the_angle_file: str, the_momentum_file: str,
                     choice: str, batch_size=20, fold: int = 5, massive: bool = False):

    """
    CALCULATING THE NUMBER OF PARAMETERS I HAVE TO DEFINE FOR THE QGNN
    """

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
    for i in dataset.dataset[0][0].nodes:  # here I use self.dataset (return a list of tuples:(nx.graph,output))
        #  attribute of FeynmanDiagramDataset class for counting the initial, final and propagator nodes
        if dataset.dataset[0][0].nodes[i]['state'][0] == 1:  # here I count initial-state nodes
            init += 1
        elif dataset.dataset[0][0].nodes[i]['state'][2] == 1:  # here I count final-state nodes
            fin += 1
        else:  # here I count propagator-state nodes
            propagator += 1
    m = int((init + fin + (propagator - 1) / 2) * propagator)  # number of combinations I can connect nodes
    n = len(dataset.dataset[0][0].nodes)  # total number of nodes

    if choice == 'unparametrized':
        obs_params = 2  # number of parameters of the observable
        total_num_params = n_layers * (m + n + kinetic_num) + obs_params # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

    elif choice == 'parametrized':
        l = len(dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        obs_params = 2  # number of parameters of the observable
        total_num_params = n_layers * (m + n + kinetic_num) + l + obs_params  # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

    elif choice == 'fully_parametrized':
        l = len(dataset.dataset[0][0].edges[(0, 2)])  # number of parameters for the feature map
        obs_params = 2  # number of parameters of the observable
        total_num_params = n_layers * (m + n + kinetic_num) + 3 * l + obs_params # KINETIC_NUM = 1 IF MASSLESS REGIME, KINETIC_NUM = 2 IF MASSIVE REGIME

    # Splitting the entire dataset into training set and test set
    cross_set, test_set = train_test_split(dataset, train_size=0.8, shuffle=True)

    dim_fold = len(cross_set)//fold

    train_loss = [0]*fold
    val_loss = [0]*fold
    final_params = [0]*fold

    the_p_stat = [0]*fold

    # doing a k-fold cross validation
    for i in range(fold):
        # Splitting q_dataset into training and validation set
        validation_set = cross_set[i*dim_fold:(i+1)*dim_fold]
        training_set = []
        for elem in cross_set:
            if elem not in validation_set:
                training_set.append(elem)

        # min-max scaling part
        _, the_p_stat[i] = min_max(training_set, validation_set)

        # Building DataLoaders for each set
        training_loader = DataLoader(training_set, batch_size=batch_size)
        validation_loader = DataLoader(validation_set)

        init_params = 0.01*torch.randn(total_num_params, dtype=torch.float)
        init_params.requires_grad = True
        print(init_params)

        final_params[i], train_loss[i], val_loss[i] = train_qgnn(training_loader, validation_loader, init_params,
                                                                 max_epoch, n_layers, choice, massive=massive)

        print("finished the ", (i+1), " cross validation process")
        print('---------------------------------------------------------------------')

    cross_train_loss = []
    cross_train_loss_std = []
    for j in range(len(train_loss[0])):
        mean = 0
        mean2 = 0
        for i in range(fold):
            mean = mean + train_loss[i][j]/fold
            mean2 = mean2 + train_loss[i][j]*train_loss[i][j]/fold

        cross_train_loss.append(mean)
        cross_train_loss_std.append(np.sqrt(mean2 - mean*mean))

    cross_train_loss = np.array(cross_train_loss)
    cross_train_loss_std = np.array(cross_train_loss_std)

    print('average loss per element after the training process:', cross_train_loss[-1]/batch_size, ' +- ', cross_train_loss_std[-1]/batch_size)

    cross_val_loss = []
    cross_val_loss_std = []
    for j in range(len(val_loss[0])):
        mean = 0
        mean2 = 0
        for i in range(fold):
            mean = mean + val_loss[i][j] / fold
            mean2 = mean2 + val_loss[i][j] * val_loss[i][j] / fold

        cross_val_loss.append(mean)
        cross_val_loss_std.append(np.sqrt(mean2-mean*mean)*fold/(fold-1))

    cross_val_loss = np.array(cross_val_loss)
    cross_val_loss_std = np.array(cross_val_loss_std)

    print('average validation loss per element at the end of the training:', cross_val_loss[-1]/len(validation_set), ' +- ', cross_val_loss_std[-1]/len(validation_set))

    # saving the loss value for each epoch
    np.savetxt(the_train_file, cross_train_loss)
    np.savetxt(the_val_file, cross_val_loss)
    np.savetxt(the_train_std_file, cross_train_loss_std)
    np.savetxt(the_val_std_file, cross_val_loss_std)

    # plotting the loss value for each epoch
    # plt.plot(range(max_epoch), cross_train_loss, color='#CC4F1B')
    # plt.fill_between(range(max_epoch), cross_train_loss - cross_train_loss_std, cross_train_loss + cross_train_loss_std,
    #                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Loss per Epoch')
    # plt.title('Loss per epoch - training set ')
    # plt.show()
    # plt.plot(range(max_epoch), cross_val_loss, color='#1B2ACC')
    # plt.fill_between(range(max_epoch), cross_val_loss - cross_val_loss_std, cross_val_loss + cross_val_loss_std,
    #                 alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Loss per Epoch')
    # plt.title('Loss per epoch - validation set ')
    # plt.show()

    final_validation = []
    for i in val_loss:
        final_validation.append(i[-1])
        optimal = min(final_validation)
    index = final_validation.index(optimal)

    print(final_params[index])
    test_loader = DataLoader(test_set)

    test_prediction(test_loader, final_params[index], the_test_file, the_truth_file, the_angle_file, the_momentum_file,
                    n_layers, choice, massive)

    print('----------------------------------------------------------------------')

    # cross_loader = DataLoader(cross_set, batch_size=batch_size)
    # init_params = 0.01*torch.randn(total_num_params, dtype=torch.float)
    # init_params.requires_grad = True
    # test_params, _, _ = train_qgnn(cross_loader, test_loader, init_params, max_epoch, n_layers, choice, massive=massive)
    # test_prediction(test_loader, test_params, the_test_file, the_truth_file, the_angle_file, the_momentum_file,
    #                n_layers, choice, massive)

    test_params = final_params[index]
    return test_params