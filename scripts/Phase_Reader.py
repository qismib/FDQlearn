from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, global_phase
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

num_layers = [3, 5]
elements = 500  # number of elements to study, I have to put it in lines 26 and 31 in FeynmanDiagramDataset
file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_t.csv'
s_phase_file = '../data/interference/s_global_phases.txt'
t_phase_file = '../data/interference/t_global_phases.txt'

feature_map = 'parametrized'  # Must be either "parametrized" or "unparametrized", it indicates the
# kind of feature map to use in training

s_array = np.loadtxt('../data/interference/parametrized_channel_s_final_params.txt')
s_params = torch.tensor(s_array, dtype=torch.float)
t_array = np.loadtxt('../data/interference/parametrized_channel_t_final_params.txt')
t_params = torch.tensor(t_array, dtype=torch.float)

torch.manual_seed(68459)
np.random.seed(68459)
# s_stat = torch.from_numpy(np.loadtxt('../data/interference/parametrized_channel_s_standardization.txt'))
s_channel = FeynmanDiagramDataset(the_file_path=file1)
# standard_scaling(s_channel, s_stat[0], s_stat[1], s_stat[2], s_stat[3], s_stat[4])
torch.manual_seed(68459)
np.random.seed(68459)
# t_stat = torch.from_numpy(np.loadtxt('../data/interference/parametrized_channel_t_standardization.txt'))
t_channel = FeynmanDiagramDataset(the_file_path=file2)
# standard_scaling(t_channel, t_stat[0], t_stat[1], t_stat[2], t_stat[3], t_stat[4])

s_channel = DataLoader(s_channel, batch_size=1)
t_channel = DataLoader(t_channel, batch_size=1)

s_phases = []
t_phases = []

for s, t in zip(s_channel, t_channel):
    s_element = (to_networkx(data=s[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                             edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), s[1][0])
    t_element = (to_networkx(data=t[0][0], graph_attrs=['scattering', 'p_norm', 'theta'], node_attrs=['state'],
                             edge_attrs=['mass', 'spin', 'charge'], to_undirected=True), t[1][0])

    s_phases.append(global_phase(s_element, s_params, num_layers[0], feature_map))
    t_phases.append(global_phase(t_element, t_params, num_layers[1], feature_map))

np.savetxt(s_phase_file, s_phases)
np.savetxt(t_phase_file, t_phases)