import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset
from qgraph import interference, matrix_squared
from torch_geometric.loader import DataLoader
from FDQLearn_main import main


def function(theta):
    q_e = np.sqrt(4*np.pi/137)
    return q_e**4*(1+np.cos(theta))**2/(2*(1-np.cos(theta)))

def bhabha(theta):
    q_e = np.sqrt(4 * np.pi / 137)
    return q_e**4*(8/(1-np.cos(theta))**2 -(1-np.cos(theta))/4 + (1+np.cos(theta))**4/(2-2*np.cos(theta))**2)


num_layers = 3
num_epoch = 30
batch = 20
elements = 500
file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_t.csv'

feature_map = 'parametrized'  # Must be either "parametrized" or "unparametrized", it indicates the
# kind of feature map to use in training

torch.manual_seed(12345)
np.random.seed(12345)
s_params = main(num_layers, num_epoch, file1, feature_map, batch, elements)

torch.manual_seed(12345)
np.random.seed(12345)
t_params = main(num_layers, num_epoch, file2, feature_map, batch, elements)

torch.manual_seed(12345)
np.random.seed(12345)
s_channel = FeynmanDiagramDataset(the_file_path=file1)
torch.manual_seed(12345)
np.random.seed(12345)
t_channel = FeynmanDiagramDataset(the_file_path=file2)

s_channel = DataLoader(s_channel, batch_size=1)
t_channel = DataLoader(t_channel, batch_size=1)


interf, angles = interference(s_channel, s_params, t_channel, t_params, num_layers, feature_map)

x = np.linspace(0.5, np.pi, 1000)
y = function(x)

plt.plot(angles, interf, 'ro')
plt.plot(x, y)
plt.show()

m_squared, angles = matrix_squared(s_channel, s_params, t_channel, t_params, num_layers, feature_map)
z = bhabha(x)

plt.plot(angles, m_squared)
plt.plot(x, z)
plt.show()