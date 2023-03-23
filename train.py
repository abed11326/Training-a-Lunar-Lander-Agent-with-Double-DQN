import gymnasium as gym
import numpy as np
import torch
from torch.nn import MSELoss
import copy
from agent import Agent
from dataStructures import ReplayBuffer, History
from hypPar import *

"""Train a neural network agent on the "Lunar Lander" simulation with DoubleDQN algorithm and saving the agent's parameters and learning curve."""

agent = Agent(state_dim, n_hidden, n_actions, initial_eps, final_eps)
replay_buffer = ReplayBuffer(buffer_size)
loss_fn = MSELoss()
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
target_network = copy.deepcopy(agent)
target_network.load_state_dict(agent.state_dict())
history = History()
counter = 0 

for episode in range(no_episodes):
    env = gym.make('LunarLander-v2', render_mode=None)
    state = env.reset()[0]
    history.reset_currentRew()
    history.reset_currentLoss()
    done = False

    while(not done):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.append([state, action, reward, next_state, done])
            
        if replay_buffer.get_len() >= batch_size:
            # This is the "learning" part. Sample data, define input and target, and optimize.
            states, actions, rewards, next_states, finished = replay_buffer.sample(batch_size)
        
            q_pred = agent.forward(states)
            targets_arr = q_pred.detach().clone() 

            with torch.no_grad():
                q_next_pred = agent.forward(next_states)
                argmax_q_next = torch.argmax(q_next_pred, dim=1)
                target_net_pred = target_network.forward(next_states).detach()
                max_q_next = target_net_pred[np.arange(batch_size), argmax_q_next]
                targets = rewards + (1-finished) * discount * max_q_next
                
            targets_arr[np.arange(batch_size), actions] = targets 
            loss = loss_fn(q_pred, targets_arr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            counter += 1
            if counter % target_update_freq == 0:
                target_network.load_state_dict(agent.state_dict())
                torch.save(agent.state_dict(), './parameters/agent_param.pt')
                print("Saved Model Parameters")
                
            history.append_step_loss(loss.data.numpy().copy())
                                            
        history.append_step_reward(reward)
        state = next_state
            
    history.update_rewHist()
    history.update_lossHist()
    print("Episode: %d,   Mean Reward: %0.3lf,   Mean Loss: %0.3lf,   Eps: %0.3lf"%(episode, history.get_last_reward(), history.get_last_loss(), agent.eps))
        
    agent.decay_eps(eps_decay)    
    env.close()
    
torch.save(agent.state_dict(), './parameters/agent_param.pt')
print("Final Saving")
history.save(0.5)