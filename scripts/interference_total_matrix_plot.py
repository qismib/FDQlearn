import matplotlib.pyplot as plt
from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset
from qgraph import interference, matrix_squared
from torch_geometric.loader import DataLoader


def function(theta):
    q_e = np.sqrt(4*np.pi/137)
    return q_e**4*(1+np.cos(theta))**2/(2*(1-np.cos(theta)))


def bhabha(theta):
    q_e = np.sqrt(4 * np.pi / 137)
    return q_e**4*(8/(1-np.cos(theta))**2 -(1-np.cos(theta))/4 + (1+np.cos(theta))**4/(2-2*np.cos(theta))**2)


interf = np.loadtxt('../data/interference/interference_outcomes.txt')
angles_1 = np.loadtxt('../data/interference/interference_angles.txt')

x = np.linspace(0.5, np.pi, 1000)
y = function(x)

plt.plot(angles_1, interf, 'ro')
plt.plot(x, y)
plt.show()

m_squared = np.loadtxt('../data/interference/total_matrix_outcomes.txt')
angles_2 = np.loadtxt('../data/interference/total_matrix_angles.txt')

z = bhabha(x)

plt.plot(angles_2, m_squared)
plt.plot(x, z)
plt.show()
