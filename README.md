# Apprenticeship Learning using Inverse Reinforcement Learning

This repository contains implementations of Deep Q-Learning (DQN) and Q-Learning algorithms for apprenticeship learning, based on the paper “Apprenticeship Learning via Inverse Reinforcement Learning" by  P. Abbeel and A. Y. Ng, applied to two classic control tasks: CartPole and Pendulum. 

## Introduction:

Apprenticeship Learning via Inverse Reinforcement Learning combines principles of reinforcement learning and inverse reinforcement learning to enable agents to learn from expert demonstrations. The agent learns to perform a task by observing demonstrations provided by an expert, without explicit guidance or reward signals. Instead of learning directly from rewards, the algorithm seeks to infer the underlying reward function from the expert demonstrations and then optimize the agent's behavior based on this inferred reward function. 

One approach to implementing this is the Projection Method Algorithm, which iteratively refines the agent's policy based on the difference between the expert's behavior and the agent's behavior. At each iteration, the algorithm computes a weight vector that maximally separates the expert's feature expectations from the agent's feature expectations, subject to a constraint on the norm of the weight vector. This weight vector is then used to train the agent's policy, and the process repeats until convergence. At least one of the trained apprentices perform at least as well as the expert within ϵ.


## CartPole

### Q-Learning

* Since the states are continuous, they are discretized into 14641 state combinations.
* Expert is trained using Q Learning algorithm with a Q-Table of dimension 14641 x 2 for 30,000 iterations within which the expert obtains the goal.

<p align="center">
  <img src="Results/Q%20Learning%20-%20CartPole/Expert%20Performance.png" width="300" />
  <img src="Results/Q%20Learning%20-%20CartPole/Expert%20Policy.gif" width="350"/>
  <p align="center">CartPole expert trained using Q learning</p>
</p>

#### Apprentices
<p align="center">
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice_1_Performance.png" width="250"/>
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice_2_Performance.png" width="250"/>
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice_4_Performance.png" width="250"/>

  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice%204%20Policy.gif" width="250"/>
</p>

<p align="center">
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice_6_Performance.png" width="250" />
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice_7_Performance.png" width="250"/>
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice_10_Performance.png" width="250"/>

  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice%206%20Policy.gif" width="250"/>
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice%207%20Policy.gif" width="250"/>
  <img src="Results/Q%20Learning%20-%20CartPole/Apprentice%2010%20Policy.gif" width="250"/>
</p>

### Deep Q-Learning

* We utilize Double Deep Q Learning to train an expert cartpole agent for 400 iterations.
* The policy is represented by the Q-value function learned by the neural network.
* A policy network and target network is used in the Double DQN.

<p align="center">
  <img src="Results/DQN%20-%20CartPole/Expert%20Performance.png" width="300" />
  <img src="Results/DQN%20-%20CartPole/Expert%20Policy.gif" width="350"/>
  <p align="center">CartPole expert trained using DDQN</p>
</p>

#### Apprentices
<p align="center">
  <img src="Results/DQN%20-%20CartPole/Apprentice_1 Performance.png" width="250"/>
  <img src="Results/DQN%20-%20CartPole/Apprentice_2 Performance.png" width="250"/>
  <img src="Results/DQN%20-%20CartPole/Apprentice_5 Performance.png" width="250"/>

  <img src="Results/DQN%20-%20CartPole/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/DQN%20-%20CartPole/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/DQN%20-%20CartPole/Apprentice%205%20Policy.gif" width="250"/>
</p>

<p align="center">
  <img src="Results/DQN%20-%20CartPole/Apprentice_7 Performance.png" width="250" />
  <img src="Results/DQN%20-%20CartPole/Apprentice_8 Performance.png" width="250"/>
  <img src="Results/DQN%20-%20CartPole/Apprentice_10 Performance.png" width="250"/>

  <img src="Results/DQN%20-%20CartPole/Apprentice%207%20Policy.gif" width="250"/>
  <img src="Results/DQN%20-%20CartPole/Apprentice%208%20Policy.gif" width="250"/>
  <img src="Results/DQN%20-%20CartPole/Apprentice%2010%20Policy.gif" width="250"/>
</p>

## Pendulum

### Q-Learning

* Since both the states and actions are continuous, they are discretized into [21, 21, 65] state combinations and 9 actions.
* Expert is trained using Q Learning algorithm for 40,000 iterations within which the expert obtains the goal.

<p align="center">
  <img src="Results/Q%20Learning%20-%20Pendulum/Expert%20Performance.png" width="323" />
  <img src="Results/Q%20Learning%20-%20Pendulum/Expert%20Policy.gif" width="250"/>
  <p align="center">Pendulum expert trained using Q learning</p>
</p>

#### Apprentices
<p align="center">
  <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice_1_Performance.png" width="250"/>
  <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice_2_Performance.png" width="250"/>
  <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice_3_Performance.png" width="250"/>

  <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice%203%20Policy.gif" width="250"/>
</p>

<p align="center">
  <p align="center">
    <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice_4_Performance.png" width="250" />
    <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice_5_Performance.png" width="250"/>
  </p>
  <p align="center">
    <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice%204%20Policy.gif" width="250"/>
    <img src="Results/Q%20Learning%20-%20Pendulum/Apprentice%205%20Policy.gif" width="250"/>
  </p>
</p>

### Deep Q-Learning

* The continuous actions are discretized into 9 actions.
* Expert agent is trained using Double Deep Q Learning for 300 iterations.

<p align="center">
  <img src="Results/DQN%20-%20Pendulum/Expert%20Performance.png" width="323" />
  <img src="Results/DQN%20-%20Pendulum/Expert%20Policy.gif" width="250"/>
  <p align="center">Pendulum expert trained using DDQN</p>
</p>

#### Apprentices
<p align="center">
  <img src="Results/DQN%20-%20Pendulum/Apprentice_1 Performance.png" width="250"/>
  <img src="Results/DQN%20-%20Pendulum/Apprentice_2 Performance.png" width="250"/>
  <img src="Results/DQN%20-%20Pendulum/Apprentice_3 Performance.png" width="250"/>

  <img src="Results/DQN%20-%20Pendulum/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/DQN%20-%20Pendulum/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/DQN%20-%20Pendulum/Apprentice%203%20Policy.gif" width="250"/>
</p>

<p align="center">
  <img src="Results/DQN%20-%20Pendulum/Apprentice_6 Performance.png" width="250" />
  <img src="Results/DQN%20-%20Pendulum/Apprentice_8 Performance.png" width="250"/>
  <img src="Results/DQN%20-%20Pendulum/Apprentice_10 Performance.png" width="250"/>

  <img src="Results/DQN%20-%20Pendulum/Apprentice%206%20Policy.gif" width="250"/>
  <img src="Results/DQN%20-%20Pendulum/Apprentice%208%20Policy.gif" width="250"/>
  <img src="Results/DQN%20-%20Pendulum/Apprentice%2010%20Policy.gif" width="250"/>
</p>


## Documentation

For an overview of the project and its implementation, refer to the [presentation](Docs/Apprenticeship%20Learning%20via%20IRL.pdf) file.

## References:
- Abbeel, P. & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning.
- Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, & Wojciech Zaremba. (2016). OpenAI Gym.
- Reinforcement Learning (DQN) Tutorial - PyTorch Tutorials 2.2.1+cu121 documentation. (n.d.).
- Amit, R., & Mataric, M. (2002). Learning movement sequences from demonstration. Proc. ICDL.
- Rhklite. (n.d.). Rhklite/apprenticeship_inverse_rl: Apprenticeship learning with inverse reinforcement learning.
- BhanuPrakashPebbeti. (n.d.). Bhanuprakashpebbeti/q-learning_and_double-Q-Learning.
- JM-Kim-94. (n.d.). JM-Kim-94/RL-Pendulum: Open ai gym - pendulum-V1 reinforcement learning (DQN, SAC).
