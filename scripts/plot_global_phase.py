import matplotlib.pyplot as plt
from pennylane import numpy as np

s_phase = np.loadtxt("../data/interference/s_global_phases.txt")
t_phase = np.loadtxt("../data/interference/t_global_phases.txt")

plt.plot(s_phase, 'ro', label="global phase for s-channel")
plt.grid()
plt.show()

plt.plot(t_phase, 'bs', label="global phase for t-channel")
plt.grid
plt.show()