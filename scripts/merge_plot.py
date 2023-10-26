from pennylane import numpy as np
import matplotlib.pyplot as plt
from qgraph import relative_error, mse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

massive = False

train_file = '../data/training_test_results/merged_training/parametrized_total_train_loss.txt'

val_s_file = '../data/training_test_results/merged_training/parametrized_total_val_s_loss.txt'

val_t_file = '../data/training_test_results/merged_training/parametrized_total_val_t_loss.txt'

train_std_file = '../data/training_test_results/merged_training/parametrized_error_train_loss.txt'

val_s_std_file = '../data/training_test_results/merged_training/parametrized_error_val_s_loss.txt'

val_t_std_file = '../data/training_test_results/merged_training/parametrized_error_val_t_loss.txt'

s_pred_file = '../data/training_test_results/merged_training/test_outcomes_e_e_s.txt'

t_pred_file = '../data/training_test_results/merged_training/test_outcomes_e_e_t.txt'

s_truth_file = '../data/training_test_results/merged_training/test_truth_e_e_s.txt'

t_truth_file = '../data/training_test_results/merged_training/test_truth_e_e_t.txt'

s_angle_file = '../data/training_test_results/merged_training/angles_e_e_s.txt'

t_angle_file = '../data/training_test_results/merged_training/angles_e_e_t.txt'

train_loss = np.loadtxt(train_file)
train_loss_std = np.loadtxt(train_std_file)
s_val_loss = np.loadtxt(val_s_file)
s_val_loss_std = np.loadtxt(val_s_std_file)
t_val_loss = np.loadtxt(val_t_file)
t_val_loss_std = np.loadtxt(val_t_std_file)

# plotting the loss value for each epoch
plt.plot(range(len(train_loss)), train_loss, color='#CC4F1B')
plt.fill_between(range(len(train_loss)), train_loss - train_loss_std, train_loss + train_loss_std,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss per Epoch')
plt.title('Loss per epoch - training set ')
plt.show()
plt.plot(range(len(s_val_loss)), s_val_loss, color='#1B2ACC')
plt.fill_between(range(len(s_val_loss)), s_val_loss - s_val_loss_std, s_val_loss + s_val_loss_std,
                 alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss per Epoch')
plt.title('Loss per epoch - s-channel validation set ')
plt.show()
plt.plot(range(len(t_val_loss)), t_val_loss, color='#00FF00')
plt.fill_between(range(len(t_val_loss)), t_val_loss - t_val_loss_std, t_val_loss + t_val_loss_std,
                 alpha=0.5, edgecolor='#00FF00', facecolor='#99FF99')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss per Epoch')
plt.title('Loss per epoch - t-channel validation set ')
plt.show()

s_truth = np.loadtxt(s_truth_file)
s_pred = np.loadtxt(s_pred_file)
s_angles = np.loadtxt(s_angle_file)

s_rel_error = relative_error(s_pred, s_truth)
print('the relative error per element for s-channel is:', s_rel_error)
s_mse = mse(s_pred, s_truth)
print('the mean squared error per element for s-channel is:', s_mse)

t_truth = np.loadtxt(t_truth_file)
t_pred = np.loadtxt(t_pred_file)
t_angles = np.loadtxt(t_angle_file)

t_rel_error = relative_error(t_pred, t_truth)
print('the relative error per element for t-channel is:', t_rel_error)
t_mse = mse(t_pred, t_truth)
print('the mean squared error per element for t-channel is:', t_mse)

# plotting lines
plt.plot(s_angles, s_pred, 'ro', label='predictions')
plt.plot(s_angles, s_truth, 'bs', label='ground truth')
plt.title('|M|^2 prediction for s-channel')
plt.xlabel('Scattering Angle (rad)')
plt.ylabel('Squared Matrix Element')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# plotting lines
plt.plot(t_angles, t_pred, 'ro', label='predictions')
plt.plot(t_angles, t_truth, 'bs', label='ground truth')
plt.title('|M|^2 prediction for t-channel')
plt.xlabel('Scattering Angle (rad)')
plt.ylabel('Squared Matrix Element')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()