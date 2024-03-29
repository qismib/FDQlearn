import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, mse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def function(theta, p):
    q_e = np.sqrt(4*np.pi/137)
    return q_e**4*(1+np.cos(theta))**2/(2*(1-np.cos(theta)))


file1 = '../data/dataset/QED_data_e_annih_e_s.csv'
file2 = '../data/dataset/QED_data_e_annih_e_t.csv'
interference_file = '../data/interference/bhabha_interference_outcomes.txt'
angle_file = '../data/interference/bhabha_angles.txt'
loss_file = '../data/interference/bhabha_interference_loss.txt'
std_file = '../data/interference/bhabha_interference_std.txt'

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
truth = function(angles, p)
loss = np.loadtxt(loss_file)
# loss_std = np.loadtxt(std_file)

a_mse = mse(predictions, truth)
print('the mean squared error per element of the test set is:', a_mse)


plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.plot(x, y, label='theoretical result')
plt.legend(loc='upper right')
plt.show()

plt.plot(angles, predictions, 'ro', label='circuit prediction')
plt.legend(loc='upper right')
plt.show()

plt.plot(range(len(loss)), loss)
# plt.fill_between(range(len(loss)), loss - loss_std, loss + loss_std,
#                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss per Epoch')
plt.title('Loss per epoch - training set ')
# plt.yscale('log')
plt.show()