import gymnasium as gym
import torch
from agent import Agent
from hypPar import *

"""Visually evaluate the agent's behaviour by loading the learned parameters and inspecting it on the simulator."""

agent = Agent(state_dim, n_hidden, n_actions, eval_eps, eval_eps)

try:
    agent.load_state_dict(torch.load('./parameters/agent_param.pt'))
    print("Loaded a saved agent")
except:
    print("Initialized agent randomly")

agent.eval()
    
for episode in range(no_episodes_eval):
    env = gym.make('LunarLander-v2', render_mode="human")
    state = env.reset()[0]
    done = False

    while(not done):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated 
        state = next_state

    env.close()