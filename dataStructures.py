import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils import exp_moving_avg
from datetime import datetime

class ReplayBuffer:
    """Stores the transitions that the algorithm would sample from."""
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def append(self, transition):
        self.buffer.append(transition)
        
    def get_len(self):
        return len(self.buffer)
    
    def sample(self, sample_size):
        idxs = np.random.choice(len(self.buffer), sample_size)
        batch = [self.buffer[idx] for idx in idxs]
        states, actions, rewards, next_states, finished = map(np.array, zip(*batch))
        states = torch.Tensor(states)
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        finished = torch.Tensor(finished)
        return states, actions, rewards, next_states, finished


class History:
    """Tracks the history of the average rewards and losses for each episode during training and saves a final learning curve."""
    def __init__(self):
        self.rewHist = []
        self.lossHist = []
        self.currentRew = []
        self.currentLoss = []

    def reset_currentRew(self):
        self.currentRew = []
    
    def reset_currentLoss(self):
        self.currentLoss = []

    def append_step_reward(self, reward):
        self.currentRew.append(reward)

    def append_step_loss(self, loss):
        self.currentLoss.append(loss)

    def update_rewHist(self):
        if len(self.currentRew) > 0:
            mean_reward = sum(self.currentRew) / len(self.currentRew)
        else:
            mean_reward = 0
        self.rewHist.append(mean_reward)
        self.reset_currentRew()

    def update_lossHist(self):
        if len(self.currentLoss) > 0:
            mean_loss = sum(self.currentLoss) / len(self.currentLoss)
        else:
            mean_loss = 0
        
        self.lossHist.append(mean_loss)
        self.reset_currentLoss()
    
    def get_last_reward(self):
        if len(self.rewHist) > 0:
            return self.rewHist[-1]
        elif len(self.currentRew) > 0:
            return sum(self.currentRew) / len(self.currentRew)
        return 0

    def get_last_loss(self):
        if len(self.lossHist) > 0:
            return self.lossHist[-1]
        elif len(self.currentLoss) > 0:
            return sum(self.currentLoss) / len(self.currentLoss)
        return 0 

    def save(self, movAvgConst):
        # plot exponentially moving average of rewards and losses
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,18))
        fig.subplots_adjust(hspace=0.3)

        rewHist = exp_moving_avg(np.array(self.rewHist), beta=movAvgConst)
        lossHist = exp_moving_avg(np.array(self.lossHist), beta=movAvgConst)
        ax1.plot(rewHist)
        ax2.plot(lossHist)

        ax1.set_title('Average reward per episode', fontsize=30)
        ax1.set_xlabel("Episode", fontsize=25)
        ax1.set_ylabel("Average reward", fontsize=25)
        ax1.grid(True)

        ax2.set_title('Average loss per episode', fontsize=30)
        ax2.set_xlabel("Episode", fontsize=25)
        ax2.set_ylabel("Average loss", fontsize=25)
        ax2.grid(True)
        plt.savefig('History_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.png')
