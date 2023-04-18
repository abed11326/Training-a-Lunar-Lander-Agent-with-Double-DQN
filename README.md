# Training a Lunar Lander Agent with Double DQN
## Introduction
A Lunar Lander is a vehicle that is intnded to land on the moon. It should land on the dedicated landing area, and, of course, land safely. This project trains a Deep RL agent to land the vehicle autonomously by learning and improving its behaviour based on a set of rewards given by the environment. <br>
## Algorithm
The vehicle is trained by a Double DQN which uses a neural network to estimate a state's action-value function for a Reinforcement Learning algorithm. The agent is implemented in [this file](agent.py), and the trained model's parameters are saved [here](parameters/agent_param.pt). It is trianed on the OpenAI Gym Lunar Lander simulator. More information about Double DQN can be found in the [original paper](https://arxiv.org/abs/1509.06461).
## Training
The main training loop can be found in [this file](train.py). The hyperparameters are defined [here](hypPar).
### Evaluation 
[This plot](History_2023-03-23-15-34-50.png) shows the learning curve of the agent. The first part shows the average reward and the second part shows the average loss per a training episode. [This file](evaluate.py) implements a visual evaluation where the agent's behaviour in the environment is observed and recorded in this [video](https://drive.google.com/file/d/1t79nWRymQDJEMdoeFJ_91TJXW-HYn0b_/view?usp=sharing).
