"""A list of all the hyperparameters required for training and evaluation."""
state_dim = 8
n_hidden = 150
n_actions = 4
buffer_size = 50000
batch_size = 128
no_episodes = 800
no_episodes_eval = 10
target_update_freq = 3000
loss_update_freq = 400 
lr = 0.00002
discount = 1.0
initial_eps = 1.0
final_eps = 0.25
eps_decay = 0.001
eval_eps = 0.01 