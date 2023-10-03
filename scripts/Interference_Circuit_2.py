import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, training_interference, interference_test, one_data_training
from qgraph import interference_gauge_setting
from torch_geometric.loader import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

num_layers = [3, 5]
elements = 300  # number of elements to study, I have to put it in lines 26 and 31 in FeynmanDiagramDataset
epochs = 50
massive_regime = False

feature_map = 'parametrized'  # Must be either "parametrized" or "unparametrized", it indicates the
# kind of feature map to use in training

file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_t.csv'
interference_file = '../data/interference/interference_outcomes.txt'

s_array = np.loadtxt('../data/interference/parametrized_channel_s_final_params.txt')
s_params = torch.tensor(s_array, dtype=torch.float, requires_grad=False)
t_array = np.loadtxt('../data/interference/parametrized_channel_t_final_params.txt')
t_params = torch.tensor(t_array, dtype=torch.float, requires_grad=False)

torch.manual_seed(68459)
np.random.seed(68459)
# s_stat = torch.from_numpy(np.loadtxt('../data/interference/parametrized_channel_s_standardization.txt'))
s_channel = FeynmanDiagramDataset(the_file_path=file1, the_n_elements=elements)
# standard_scaling(s_channel, s_stat[0], s_stat[1], s_stat[2], s_stat[3], s_stat[4])
torch.manual_seed(68459)
np.random.seed(68459)
# t_stat = torch.from_numpy(np.loadtxt('../data/interference/parametrized_channel_t_standardization.txt'))
t_channel = FeynmanDiagramDataset(the_file_path=file2, the_n_elements=elements)
# standard_scaling(t_channel, t_stat[0], t_stat[1], t_stat[2], t_stat[3], t_stat[4])

s_channel = DataLoader(s_channel, batch_size=1)
t_channel = DataLoader(t_channel, batch_size=1)

init_params = 0.01*torch.randn(2, dtype=torch.float)
init_params.requires_grad = True

print("i parametri iniziali sono:", init_params)

# init_params = training_interference(s_channel, s_params, t_channel, t_params, init_params, epochs, num_layers,
                                    # feature_map, massive=massive_regime)

# predictions, ground_truth, angles = interference_test(s_channel, s_params, t_channel, t_params, init_params,
                                                      # num_layers, feature_map, massive=massive_regime)

predictions, ground_truth, angles = one_data_training(s_channel, s_params, t_channel, t_params, epochs, num_layers,
                                                      feature_map, massive=massive_regime)

# predictions, ground_truth, angles, loss = interference_gauge_setting(s_channel, s_params, t_channel, t_params,
                                                                     # num_layers, feature_map, massive=massive_regime)

print('i parametri finali sono:',init_params)

predictions = [p.detach().numpy() for p in predictions]

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.plot(angles, ground_truth, 'bs', label='theoretical result')
plt.legend(loc='upper right')
plt.show()

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.legend(loc='upper right')
plt.show()

# plt.plot(range(len(loss)), loss, 'ro', label='squared-error of the prediction')
# plt.legend(loc='upper right')
# plt.show()