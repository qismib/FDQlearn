from pennylane import numpy as np
import torch
from qgraph import FeynmanDiagramDataset, model_evaluation


# fixing the seeds:
torch.manual_seed(12345)
np.random.seed(12345)

num_layers = 3
k_fold = 5
num_epoch = 50
batch = 20
elements = 1000
massive_regime = True

csv_file = '../data/dataset/QED_data_e_annih_mu_massive.csv'

q_dataset = FeynmanDiagramDataset(the_file_path=csv_file, the_n_elements=elements)

train_file = '../data/training_test_results/e_mu_massive/parametrized_e_mu_train_loss_massive_3l.txt'

val_file = '../data/training_test_results/s_channel_massive/parametrized_e_mu_val_loss_massive_3l.txt'

test_pred_file = '../data/training_test_results/s_channel_massive/parametrized_e_mu_predictions_massive_3l.txt'

truth_file = '../data/training_test_results/s_channel_massive/parametrized_e_mu_truth_massive_3l.txt'

final_params_file = '../data/training_test_results/s_channel_massive/parametrized_e_mu_final_params_massive_3l.txt'

feature_map = 'parametrized'  # Must be either "parametrized", "unparametrized" or "fully_connected", it indicates the
# kind of feature map to use in training

params = model_evaluation(num_layers, num_epoch, q_dataset, train_file, val_file, test_pred_file,
                          truth_file, feature_map, batch, fold=k_fold, massive=massive_regime)

params = [i.detach().numpy() for i in params]
np.savetxt(final_params_file, params)