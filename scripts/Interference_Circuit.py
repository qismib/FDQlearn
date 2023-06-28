import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset
from qgraph import interference, matrix_squared, standard_scaling
from torch_geometric.loader import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

num_layers = 3
num_epoch = 30
batch = 20
elements = 500
file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_t.csv'
interference_file = '../data/interference/interference_outcomes.txt'
total_matrix_file = '../data/interference/total_matrix_outcomes.txt'

feature_map = 'parametrized'  # Must be either "parametrized" or "unparametrized", it indicates the
# kind of feature map to use in training

s_array = np.loadtxt('../data/interference/parametrized_channel_s_final_params.txt')
s_params = torch.tensor(s_array, dtype=torch.float)
t_array = np.loadtxt('../data/interference/parametrized_channel_t_final_params.txt')
t_params = torch.tensor(t_array, dtype=torch.float)

torch.manual_seed(12345)
np.random.seed(12345)
s_stat = torch.from_numpy(np.savetxt('../data/interference/parametrized_channel_s_standardization.txt'))
s_channel = FeynmanDiagramDataset(the_file_path=file1)
standard_scaling(s_channel, s_stat[0], s_stat[1], s_stat[2], s_stat[3], s_stat[4])
torch.manual_seed(12345)
np.random.seed(12345)
t_stat = torch.from_numpy(np.savetxt('../data/interference/parametrized_channel_t_standardization.txt'))
t_channel = FeynmanDiagramDataset(the_file_path=file2)
standard_scaling(t_channel, t_stat[0], t_stat[1], t_stat[2], t_stat[3], t_stat[4])

s_channel = DataLoader(s_channel, batch_size=1)
t_channel = DataLoader(t_channel, batch_size=1)


interf, angles_1 = interference(s_channel, s_params, t_channel, t_params, num_layers, interference_file, feature_map)

np.savetxt(interference_file, interf)
np.savetxt('../data/interference/interference_angles.txt', angles_1)

m_squared, angles_2 = matrix_squared(s_channel, s_params, t_channel, t_params, num_layers, total_matrix_file, feature_map)

np.savetxt(total_matrix_file, m_squared)
np.savetxt('../data/interference/total_matrix_angles.txt', angles_2)