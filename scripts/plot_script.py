from pennylane import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_file = ['../data/training_test_results/parametrized_s_channel_train_loss_2l.txt',
              '../data/training_test_results/parametrized_s_channel_train_loss_3l.txt',
              '../data/training_test_results/parametrized_s_channel_train_loss_4l.txt',
              '../data/training_test_results/parametrized_s_channel_train_loss_5l.txt']


val_file = ['../data/training_test_results/parametrized_s_channel_val_loss_2l.txt',
            '../data/training_test_results/parametrized_s_channel_val_loss_3l.txt',
            '../data/training_test_results/parametrized_s_channel_val_loss_4l.txt',
            '../data/training_test_results/parametrized_s_channel_val_loss_5l.txt']

num_layers = [2, 3, 4, 5]

training_loss = [0]*len(num_layers)
validation_loss = [0]*len(num_layers)

for i in range(len(num_layers)):
    training_loss[i] = np.loadtxt(train_file[i])
    plt.plot(range(len(training_loss[i])), training_loss[i], label=str(i+1)+' layers training loss')
    # plt.yscale('log')
    plt.title('Training loss for different depths')
    plt.legend(loc='upper right')
plt.show()


for i in range(len(num_layers)):
    validation_loss[i] = np.loadtxt(val_file[i])
    plt.plot(range(len(validation_loss[i])), validation_loss[i], label=str(i+1) + ' layers validation loss')
    # plt.yscale('log')
    plt.title('Validation loss for different depths')
    plt.legend(loc='upper right')
plt.show()