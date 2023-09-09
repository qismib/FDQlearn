from pennylane import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

angles_e_e_s = np.loadtxt('../data/training_test_results/angles_e_e_s.txt')
angles_e_e_t = np.loadtxt('../data/training_test_results/angles_e_e_t.txt')
truth_e_e_s = np.loadtxt('../data/training_test_results/truth_e_e_s.txt')
truth_e_e_t = np.loadtxt('../data/training_test_results/truth_e_e_t.txt')
predictions_e_e_s = np.loadtxt('../data/training_test_results/outcomes_e_e_s.txt')
predictions_e_e_t = np.loadtxt('../data/training_test_results/outcomes_e_e_t.txt')
train_loss = np.loadtxt('../data/training_test_results/parametrized_total_train_loss.txt')
val_loss = np.loadtxt('../data/training_test_results/parametrized_total_val_loss.txt')
val_s_loss = val_loss[:len(val_loss)//2]
val_t_loss = val_loss[len(val_loss)//2:]


epochs = range(len(train_loss))

plt.figure(1)
plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_s_loss, label='validation s-channel loss')
plt.plot(epochs, val_t_loss, label='validation t-channel loss')
plt.show()

plt.figure(2)
plt.plot(angles_e_e_s, truth_e_e_s, 'ro')
plt.plot(angles_e_e_s, predictions_e_e_s, 'bs')
plt.show()

plt.figure(3)
plt.plot(angles_e_e_t, truth_e_e_t, 'ro')
plt.plot(angles_e_e_t, predictions_e_e_t, 'bs')
plt.show()