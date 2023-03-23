import torch
from torch.nn import Module, Linear, ReLU
import numpy as np
from utils import argmaxes

class Agent(Module):
    """The neural network agent that is trained to interact with the environment. Uses epsilon-greedy for exploration."""
    def __init__(self, state_dim, n_hidden, n_actions, initial_eps, final_eps):
        super(Agent, self).__init__()
        self.n_actions = n_actions
        self.eps = initial_eps
        self.final_eps = final_eps

        self.fc1 = Linear(state_dim, n_hidden)
        self.fc2 = Linear(n_hidden, n_hidden)
        self.fc3 = Linear(n_hidden, n_hidden)
        self.out = Linear(n_hidden, n_actions)

    def forward(self, X):
        X = self.fc1(X)
        X = ReLU()(X)
        X = self.fc2(X)
        X = ReLU()(X)
        X = self.fc3(X)
        X = ReLU()(X)
        q_vals = self.out(X)
        return q_vals

    def select_action(self, curr_state):
        if np.random.rand() < self.eps:
            selected_act = np.random.choice(self.n_actions)
        else:
            q_vals = self.forward(torch.Tensor(curr_state))
            selected_act = np.random.choice(argmaxes(q_vals.detach().numpy()))

        return selected_act

    def decay_eps(self, value):
        if self.eps > self.final_eps:
            self.eps -= value
