import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, relative_error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def function(theta, p):
    q_e = np.sqrt(4*np.pi/137)
    return q_e**4*(1+np.cos(theta))**2/(2*(1-np.cos(theta)))


file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_t.csv'
interference_file = '../data/interference/bhabha_interference_outcomes.txt'
angle_file = '../data/interference/bhabha_angles.txt'
# loss_file = '../data/interference/bhabha_interference_loss.txt'

torch.manual_seed(68459)
np.random.seed(68459)
s_channel = FeynmanDiagramDataset(the_file_path=file1, the_n_elements=1)
torch.manual_seed(68459)
np.random.seed(68459)
t_channel = FeynmanDiagramDataset(the_file_path=file2, the_n_elements=1)

p = s_channel[0][0]['p_norm'].numpy()

x = np.linspace(0.5, np.pi, 1000)
y = function(x, p)

angles = np.loadtxt(angle_file)
predictions = np.loadtxt(interference_file)
# loss = np.loadtxt(loss_file)

truth = function(angles, p)
rel_err = relative_error(predictions, truth)
print(rel_err)

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.plot(x, y, label='theoretical result')
plt.legend(loc='upper right')
plt.show()

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.legend(loc='upper right')
plt.show()

# plt.plot(len(loss), loss, label='optimization loss')
# plt.show()