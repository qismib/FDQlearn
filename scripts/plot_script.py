from pennylane import numpy as np
import matplotlib.pyplot as plt
from qgraph import relative_error, get_mse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

massive = False

train_file = '../data/training_test_results/Z0_channel/parametrized_Z0_channel_train_loss_3l.txt'

val_file = '../data/training_test_results/Z0_channel/parametrized_Z0_channel_val_loss_3l.txt'

test_pred_file = '../data/training_test_results/Z0_channel/parametrized_Z0_channel_predictions_3l.txt'

truth_file = '../data/training_test_results/Z0_channel/parametrized_Z0_channel_ground_truth_3l.txt'

train_std_file = '../data/training_test_results/Z0_channel/parametrized_Z0_channel_train_std_3l.txt'

val_std_file = '../data/training_test_results/Z0_channel/parametrized_Z0_channel_val_std_3l.txt'

angle_file = '../data/training_test_results/Z0_channel/parametrized_Z0_channel_angles_3l.txt'

momenta_file = '../data/training_test_results/Z0_channel/parametrized_Z0_momenta_3l.txt'

train_loss = np.loadtxt(train_file)
train_loss_std = np.loadtxt(train_std_file)
val_loss = np.loadtxt(val_file)
val_loss_std = np.loadtxt(val_std_file)

# plotting the loss value for each epoch
plt.plot(range(len(train_loss)), train_loss, color='#CC4F1B')
plt.fill_between(range(len(train_loss)), train_loss - train_loss_std, train_loss + train_loss_std,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss per Epoch')
plt.title('Loss per epoch - training set ')
plt.show()
plt.plot(range(len(val_loss)), val_loss, color='#CC4F1B')
plt.fill_between(range(len(val_loss)), val_loss - val_loss_std, val_loss + val_loss_std,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss per Epoch')
plt.title('Loss per epoch - validation set ')
plt.show()

truth = np.loadtxt(truth_file)
pred = np.loadtxt(test_pred_file)
angles = np.loadtxt(angle_file)

rel_error = relative_error(pred, truth)
print('the relative error per element of the test set is:', rel_error)
mse = get_mse(pred, truth) / len(truth)
print('the mean squared error per element of the test set is:', mse)

# plotting lines
plt.plot(angles, pred, 'ro', label='predictions')
plt.plot(angles, truth, 'bs', label='ground truth')
plt.title('|M|^2 prediction over the test set')
plt.xlabel('Scattering Angle (rad)')
plt.ylabel('Squared Matrix Element')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
if massive is True:
    momentum = np.loadtxt(momenta_file)
    ax = plt.axes(projection='3d')
    ax.scatter3D(angles, momentum, pred, color='red')
    ax.scatter3D(angles, momentum, truth, color='blue')
    ax.set_title('Squared Matrix Element')
    ax.set_xlabel('Scattering Angle (rad)', fontsize=12)
    ax.set_ylabel('Momentum of the particles', fontsize=12)
    ax.set_zlabel('Squared Matrix Element', fontsize=12)
    plt.show()