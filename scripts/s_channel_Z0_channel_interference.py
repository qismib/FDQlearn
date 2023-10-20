import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, training_interference, interference_test, one_data_training
from qgraph import interference_gauge_setting
from torch_geometric.loader import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def function(theta, p):
    m_Z = 91187
    theta_weinberg = 0.488
    gamma_Z = 2490
    q_e = np.sqrt(4 * np.pi / 137)
    A_e = -0.5
    V_e = -0.5 + 2 * (np.sin(theta_weinberg)) ** 2
    return 8*p**2*(4*p**2-m_Z**2)*(q_e**2/np.sin(2*theta_weinberg))**2*((1+np.cos(theta)**2)*V_e**2+2*np.cos(theta)*A_e**2)/((4*p**2-m_Z**2)**2+m_Z**2*gamma_Z**2)


num_layers = [3, 5]
elements = 100  # number of elements to study, I have to put it in lines 26 and 31 in FeynmanDiagramDataset
epochs = 200
kfold = 5
massive_regime = False

feature_map = 'parametrized'  # Must be either "parametrized" or "unparametrized", it indicates the
# kind of feature map to use in training

file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_Z.csv'
interference_file = '../data/interference/s_Z0_interference_outcomes.txt'
angles_file = '../data/interference/s_Z0_angles.txt'

s_array = np.loadtxt('../data/interference/parametrized_channel_s_final_params_3l.txt')
s_params = torch.tensor(s_array, dtype=torch.float, requires_grad=False)
z_array = np.loadtxt('../data/interference/parametrized_Z0_channel_final_params_5l.txt')
z_params = torch.tensor(z_array, dtype=torch.float, requires_grad=False)

torch.manual_seed(68459)
np.random.seed(68459)
s_channel = FeynmanDiagramDataset(the_file_path=file1, the_n_elements=elements)
torch.manual_seed(68459)
np.random.seed(68459)
z_channel = FeynmanDiagramDataset(the_file_path=file2, the_n_elements=elements)

s_channel = DataLoader(s_channel, batch_size=1)
z_channel = DataLoader(z_channel, batch_size=1)

init_params = 0.01*torch.randn(2, dtype=torch.float)
init_params.requires_grad = True

print("i parametri iniziali sono:", init_params)

# init_params = training_interference(function, s_channel, s_params, z_channel, z_params, init_params, epochs, num_layers,
                                    # feature_map, massive=massive_regime)

# predictions, angles = interference_test(s_channel, s_params, z_channel, z_params, init_params,
                                        # num_layers, feature_map, massive=massive_regime)

predictions, angles = one_data_training(function, s_channel, s_params, z_channel, z_params, epochs, num_layers,
                                        kfold, feature_map, massive=massive_regime)

# predictions, angles, loss = interference_gauge_setting(function, s_channel, s_params, z_channel, z_params,
                                                       # num_layers, feature_map, massive=massive_regime)

print('i parametri finali sono:', init_params)

predictions = [p.detach().numpy() for p in predictions]
np.savetxt(interference_file, predictions)
np.savetxt(angles_file, angles)

x = np.linspace(0.5, np.pi, 1000)
y = function(x)

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.plot(x, y, label='theoretical result')
plt.legend(loc='upper right')
plt.show()

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.legend(loc='upper right')
plt.show()