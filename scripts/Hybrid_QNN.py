import pennylane as qml
from pennylane import numpy as np
import torch
from torch import optim, nn
import torch_geometric.nn as geom
from qgraph import FeynmanDiagramDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import time


def get_mse(the_predictions, the_ground_truth):
    """
    :param the_predictions: list of predictions of the QGNN
    :param  the_ground_truth: list of true labels
    :return: mse: sum of squared errors
    """
    the_n = len(the_predictions)
    assert len(the_ground_truth) == the_n, "The number of predictions and true labels is not equal"
    return sum([(the_predictions[i] - the_ground_truth[i]) ** 2 for i in range(the_n)])

########################################################################################################################


def training(the_model, the_training_loader, the_validation_loader, the_n_epochs, the_train_file, the_val_file):
    """
    :param the_model: Hybrid QNN and its traiable parameters
    :param  the_training_loader: DataLoader object of the training set
    :param the_validation_loader: DataLoader object of the validation set
    :param  the_n_epochs: number of epochs of the training process
    :param the_train_file: file where I save the training loss function per epoch
    :param the_val_file: file where I save the validation loss function per epoch
    :return: the_weights: list of the final weights after the training
    """

    opt = optim.Adam(the_model.parameters(), lr=1e-2)  # initialization of the optimizer to use
    epoch_loss = []
    validation_loss = []

    for epoch in range(the_n_epochs):

        costs = []
        starting_time = time.time()

        for _, item in enumerate(the_training_loader):

            batch_truth = item[1]
            batch_graph = item[0]

            def opt_func():  # defining an optimization function for the training of the model

                batch_predition = the_model(batch_graph)
                loss = get_mse(batch_predition, batch_truth)
                costs.append(loss.item())
                loss.backward()
                return loss

            opt.zero_grad()
            opt.step(opt_func)

        ending_time = time.time()
        elapsed = ending_time - starting_time

        training_loss = np.mean(costs)
        epoch_loss.append(training_loss)

        the_val = validation(the_validation_loader, the_model)
        validation_loss.append(the_val)

        if epoch != 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 1e-5:
            the_n_epochs = epoch + 1  # Have to add 1 for plotting the right number of epochs
            break

        if epoch % 5 == 0:
            res = [epoch, training_loss, elapsed]
            print("Epoch: {:2d} | Training loss: {:3f} | Elapsed Time per Epoch: {:3f}".format(*res))

    # saving the loss value for each epoch
    np.savetxt(the_train_file, epoch_loss)
    np.savetxt(the_val_file, validation_loss)

    # plotting the loss value for each epoch
    plt.plot(range(the_n_epochs), epoch_loss, label='training')
    plt.plot(range(the_n_epochs), validation_loss, label='validation')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss per Epoch')
    plt.legend(loc="upper right")
    plt.show()
    return the_model.parameters()


def validation(the_validation_loader, the_model):
    """
    :param the_validation_loader: DataLoader object of the validation set
    :param the_model: hybrid quantum neural network and its weights
    :return: the_validation_loss: list of loss values for each point of the validation set after prediction
    """

    the_val_node_feat = []
    the_val_adj_list = []
    the_val_edge_feat = []
    the_val_truth = []

    for _, item in enumerate(the_validation_loader):

        y_val_hat = item[1]
        y_val_pred = the_model(item[0])

    # define a list of prediction
    the_val_predictions = the_model(the_val_node_feat, the_val_adj_list, the_val_edge_feat)
    assert len(the_val_truth) == len(the_val_predictions), "The number of predictions and true labels is not equal"

    the_validation_loss = get_mse(the_val_predictions, the_val_truth)

    return the_validation_loss.detach().numpy()


########################################################################################################################

def feature_map(the_feature_array, num_qubits):

    assert len(the_feature_array) == num_qubits, 'the number of features must be the same as the number of qubits'

    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    qml.BasicEntanglerLayers(the_feature_array, wires=range(num_qubits), rotation=qml.RZ)


def ansatz(the_num_qubits, the_n_layers, the_weights):

    for j in range(the_n_layers):
        for i in range(the_num_qubits):
            qml.RX(the_weights[i][j], wires=i)
            qml.RZ(the_weights[i][j], wires=i)

        for i in range(the_num_qubits):
            qml.CNOT(wires=(i, (i+1) % the_num_qubits))


class HybridQgnnModel(nn.Module):

    def __init__(self, the_num_nodes, the_num_edges, the_num_node_features, the_num_edge_features,
                 the_node_embedding_size, the_edge_embedding_size, n_layer: int = 3):

        global feature_map, ansatz

        super(HybridQgnnModel, self).__init__()

        self.num_nodes = the_num_nodes
        self.num_edges = the_num_edges
        self.num_node_features = the_num_node_features
        self.num_edge_features = the_num_edge_features
        self.node_embedding_size = the_node_embedding_size
        self.edge_embedding_size = the_edge_embedding_size
        self.gcn_node = geom.GCNConv(in_channels=self.num_node_features, out_channels=1)
        self.gcn_edge = geom.GCNConv(in_channels=self.num_edge_features, out_channels=1)
        self.node_embedding = nn.Linear(self.num_nodes, self.node_embedding_size)
        self.edge_embedding = nn.Linear(self.num_edges, self.edge_embedding_size)

        n_qubits = self.node_embedding_size + self.edge_embedding_size + 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method="parameter-shift")
        def pennylane_qnn(inputs, weights):
            # inputs is a list composed by: attributes_array (concatenation of all the feature AFTER convolutional
            # layer), the_num_qubits (total number of qubits of the circuit), ansatz_layers (depth of the circuit)
            feature_map(inputs[0], inputs[1])

            for i in range(inputs[2]):
                ansatz(weights[i])
            return qml.expval(qml.PauliZ(wires=0))

        self.n_qubits = n_qubits
        self.n_layer = n_layer
        self.weight_shapes = {'weights': (2, n_qubits, n_layer)}  # shape of the parameters
        self.pennylane_qnn = pennylane_qnn
        self.qnn = qml.qnn.TorchLayer(pennylane_qnn, self.weight_shapes)

    def forward(self, the_batch):
        edge_list = the_batch['edge_index']
        x_nodes = the_batch['state']
        print(x_nodes.size())
        x_edges = torch.cat((the_batch['mass'].view(-1, 1), the_batch['spin'].view(-1, 1),
                             the_batch['charge'].view(-1, 1)), dim=1)
        print(x_edges)
        print(x_edges.size())
        p = the_batch['p_norm']
        theta = the_batch['theta']
        scattering = the_batch['scattering']

        x1 = self.gcn_node(x_nodes, edge_list).view(-1, self.num_nodes)
        # x1 = torch.tensor(x1).view(-1, self.num_nodes)
        x1 = self.node_embedding(x1)
        x2 = self.gcn_edge(x_edges, edge_list).view(-1, self.num_edges)
        # x2 = torch.tensor(x2).view(-1, self.num_edges)
        x2 = self.edge_embedding(x2)
        features_array = torch.cat((x1, x2, p, theta))
        qnn_inputs = [features_array, self.n_qubits, self.n_layer]
        y = self.qnn(qnn_inputs, self.weight_shapes)
        return y

########################################################################################################################


file = '../data/dataset/QED_data_qed.csv'
train_loss = '../data/hybrid_train_test/hybrid_train.txt'
val_loss = '../data/hybrid_train_test/hybrid_val.txt'
num_elem = 750
batch_size = 20
epochs = 30

q_dataset = FeynmanDiagramDataset(the_file_path=file, the_n_elements=num_elem)

num_nodes = len(q_dataset.dataset[0][0].nodes)
num_node_attr = len(q_dataset.dataset[0][0].nodes[0]['state'])
num_edges = len(q_dataset.dataset[0][0].edges)
num_edge_attr = len(q_dataset.dataset[0][0].edges[(2, 3)])
node_embedding_size = 3
edge_embedding_size = 3
layer = 3

# Splitting q_dataset into training, test and validation set
training_set, test_set = train_test_split(q_dataset, train_size=0.8)
training_set, validation_set = train_test_split(training_set, train_size=0.8)

# Building DataLoaders for each set
training_loader = DataLoader(training_set, batch_size=batch_size)
test_loader = DataLoader(test_set)
validation_loader = DataLoader(validation_set)

model = HybridQgnnModel(num_nodes, num_edges, num_node_attr, num_edge_attr, node_embedding_size, edge_embedding_size, layer)

final_params = training(model, training_loader, validation_loader, epochs, train_loss, val_loss)

