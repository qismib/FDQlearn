import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, relative_error
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
s_channel = FeynmanDiagramDataset(the_file_path=file1, the_n_elements=1)
torch.manual_seed(68459)
np.random.seed(68459)
z_channel = FeynmanDiagramDataset(the_file_path=file2, the_n_elements=1)

p = s_channel[0][0]['p_norm'].numpy()

angles = np.loadtxt(angles_file)
predictions = np.loadtxt(interference_file)
# loss = np.loadtxt(loss_file)

x = np.linspace(0.5, np.pi, 1000)
y = function(x, p)

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.plot(x, y, label='theoretical result')
plt.legend(loc='upper right')
plt.show()

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.legend(loc='upper right')
plt.show()

# plt.plot(len(loss), loss, label='optimization loss')
# plt.show()