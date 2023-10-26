import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, training_interference, interference_test, one_data_training
from qgraph import interference_gauge_setting, relative_error
from torch_geometric.loader import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def function(theta, p):
    q_e = np.sqrt(4*np.pi/137)
    return q_e**4*(1+np.cos(theta))**2/(2*(1-np.cos(theta)))


num_layers = [3, 3]
elements = 150  # number of elements to study, I have to put it in lines 26 and 31 in FeynmanDiagramDataset
epochs = 150
kfold = 5
massive_regime = False

feature_map = 'parametrized'  # Must be either "parametrized" or "unparametrized", it indicates the
# kind of feature map to use in training

file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_t.csv'
interference_file = '../data/interference/bhabha_interference_outcomes.txt'
angle_file = '../data/interference/bhabha_angles.txt'
loss_file = '../data/interference/bhabha_interference_loss.txt'

s_array = np.loadtxt('../data/interference/parametrized_channel_s_final_params_3l.txt')
s_params = torch.tensor(s_array, dtype=torch.float, requires_grad=False)
t_array = np.loadtxt('../data/interference/parametrized_channel_t_final_params_3l.txt')
t_params = torch.tensor(t_array, dtype=torch.float, requires_grad=False)

torch.manual_seed(68459)
np.random.seed(68459)
s_channel = FeynmanDiagramDataset(the_file_path=file1, the_n_elements=elements)
torch.manual_seed(68459)
np.random.seed(68459)
t_channel = FeynmanDiagramDataset(the_file_path=file2, the_n_elements=elements)

p = s_channel[0][0]['p_norm'].numpy()

s_channel = DataLoader(s_channel, batch_size=1)
t_channel = DataLoader(t_channel, batch_size=1)

init_params = 0.01*torch.randn(2, dtype=torch.float)
init_params.requires_grad = True

print("i parametri iniziali sono:", init_params)

init_params, int_loss = training_interference(function, s_channel, s_params, t_channel, t_params, init_params, epochs, num_layers,
                                              feature_map, massive=massive_regime)
np.savetxt(loss_file, int_loss)

predictions, angles = interference_test(s_channel, s_params, t_channel, t_params, init_params,
                                        num_layers, feature_map, massive=massive_regime)

# predictions, angles = one_data_training(function, s_channel, s_params, t_channel, t_params, epochs, num_layers,
                                        # kfold, feature_map, massive=massive_regime)

# predictions, angles, loss = interference_gauge_setting(function, s_channel, s_params, t_channel, t_params,
                                                       # num_layers, feature_map, massive=massive_regime)

print('i parametri finali sono:', init_params)

predictions = [p.detach().numpy() for p in predictions]
np.savetxt(interference_file, predictions)
np.savetxt(angle_file, angles)