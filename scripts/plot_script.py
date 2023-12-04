from pennylane import numpy as np
import matplotlib.pyplot as plt
from qgraph import relative_error, mse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

massive = False

csv_file = '../data/dataset/QED_data_e_annih_e_s.csv'

train_file = '../data/training_test_results/s_channel/parametrized_s_channel_train_loss_3l.txt'

val_file = '../data/training_test_results/s_channel/parametrized_s_channel_val_loss_3l.txt'

test_pred_file = '../data/training_test_results/s_channel/parametrized_s_channel_predictions_3l.txt'

truth_file = '../data/training_test_results/s_channel/parametrized_s_channel_ground_truth_3l.txt'

final_params_file = '../data/interference/parametrized_channel_s_final_params_3l.txt'

train_std_file = '../data/training_test_results/s_channel/parametrized_s_channel_train_std_3l.txt'

val_std_file = '../data/training_test_results/s_channel/parametrized_s_channel_val_std_3l.txt'

angle_file = '../data/training_test_results/s_channel/parametrized_s_channel_angles_3l.txt'

momenta_file = '../data/training_test_results/s_channel/parametrized_s_channel_momenta_3l.txt'

train_loss = np.loadtxt(train_file)
train_loss_std = np.loadtxt(train_std_file)
val_loss = np.loadtxt(val_file)
val_loss_std = np.loadtxt(val_std_file)

# plotting the loss value for each epoch
plt.plot(range(len(train_loss)), train_loss, color='#CC4F1B')
plt.fill_between(range(len(train_loss)), train_loss - train_loss_std, train_loss + train_loss_std,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.xlabel('Number of Epochs', size=13)
plt.ylabel('Loss per Epoch', size=13)
plt.title('Loss per epoch - training set', size=15)
plt.xticks(size=10)
plt.yticks(size=10)
plt.yscale('log')
plt.show()

plt.plot(range(len(val_loss)), val_loss, color='#1B2ACC')
plt.fill_between(range(len(val_loss)), val_loss - val_loss_std, val_loss + val_loss_std,
                 alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
plt.xlabel('Number of Epochs', size=13)
plt.ylabel('Loss per Epoch', size=13)
plt.title('Loss per epoch - validation set', size=15)
plt.xticks(size=10)
plt.yticks(size=10)
plt.yscale('log')
plt.show()

print('MSE for the training set after the cross val:', train_loss[-1]/20, '+-', train_loss_std[-1]/20)
print('MSE for the validation set after the cross val:', val_loss[-1]/160, '+-', val_loss_std[-1]/160)

truth = np.loadtxt(truth_file)
pred = np.loadtxt(test_pred_file)
angles = np.loadtxt(angle_file)

rel_error = relative_error(pred, truth)
print('the relative error per element of the test set is:', rel_error)
a_mse = mse(pred, truth)
print('the mean squared error per element of the test set is:', a_mse)

# plotting lines
plt.plot(angles, pred, 'X', label='predictions')
plt.plot(angles, truth, 'r.', label='ground truth')
plt.title('|M|^2 prediction over the test set', size=15)
plt.xlabel('Scattering Angle (rad)', size=13)
plt.ylabel('Squared Matrix Element', size=13)
plt.legend(loc='lower right', fontsize=12)
plt.xticks(size=10)
plt.yticks(size=10)
plt.grid(True)
plt.show()
if massive is True:
    momentum = np.loadtxt(momenta_file)
    ax = plt.axes(projection='3d')
    ax.scatter3D(angles, momentum, pred, color='red')
    ax.scatter3D(angles, momentum, truth, color='blue')
    ax.set_title('Squared Matrix Element', size=15)
    ax.set_xlabel('Scattering Angle (rad)', size=13)
    ax.set_ylabel('Momentum', size=13)
    ax.set_zlabel('Squared Matrix Element', size=13)
    plt.show()

residuals = []

for i, j in zip(truth, pred):
    residuals.append(j - i)

residuals = np.array(residuals)

plt.plot(residuals, '-x')
plt.ylabel('residuals', size=13)
plt.xlabel('data', size=13)
plt.title("residuals for massive Bhabha s-channel", size=15)
plt.show()