import copy
from collections import deque, namedtuple
from itertools import count
import math
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
import os
from tqdm import tqdm
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

import gym


# Neural Network for Deep Q Learning (DQN)
class DQN(nn.Module):
    def __init__(self, n_observations=4, hidden_layer_size=64, n_actions=2):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(n_observations, hidden_layer_size)
        self.hidden_layer_1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.hidden_layer_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, n_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x
    
    
# Class to store the experiences (transitions) of the agent as it interacts with the environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
# Class for training agent (expert) 

class DQNTrainer:
    
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.95
    EPS_END = 0.05
    TARGET_UPDATE = 10
    resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.BICUBIC),T.ToTensor()])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self, env, agent, num_episodes = 200, save_path=None, load_path=None):
        
        self.env = env
        if save_path:
            self.env = gym.wrappers.Monitor(env, save_path, video_callable=lambda x: x % 199 == 0, force=True)
        self.env.reset()
        
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(100000)
        
        self.steps_done = 0
        self.num_episodes = num_episodes
        self.EPS_DECAY = num_episodes * 0.9
        
        self.episode_durations = []
        self.best_reward = -float('inf')
        
        self.is_trained = False
        self.best_model = None
        
        if load_path:
            try:
                self.is_trained = True
                data = torch.load(load_path)
                self.policy_net.load_state_dict(data)
                self.best_model = self.policy_net
            except (FileNotFoundError, torch.cuda.CudaError, RuntimeError, KeyError) as e:
                print(f"Error loading pretrained model: {e}")
        
        self.agent = agent
        
    def get_cart_location(self, screen_width):
        """
        Get horizontal position of the cart's center on the screen.
        """
        world_width = self.env.unwrapped.x_threshold * 2
        scale = screen_width / world_width
        cart_position_on_screen = int(self.env.unwrapped.state[0] * scale + screen_width / 2.0)
        return cart_position_on_screen


    def get_screen_image(self):
        """
        Get a pre-processed screen image from the environment for input to a neural network.
        """
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        top_cutoff, bottom_cutoff = int(screen_height * 0.4), int(screen_height * 0.8)
        screen = screen[:, top_cutoff:bottom_cutoff]

        # position of the cart on the screen
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)

        # define the range for extracting a square image centered on the cart
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

        # extract the square image centered on the cart
        screen = screen[:, :, slice_range]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        return self.resize(screen).unsqueeze(0).to(self.device)


    def show_screen_image(self):
        """
        Output pre-processed screen image image.
        """
        plt.figure()
        plt.axis('off')
        plt.imshow(self.get_screen_image().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
        plt.title('Preprocessed Screen Image')


    def select_action(self, state):
        """
        Select an action according to an epsilon greedy policy.
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.inference_mode():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)


    def plot_progress(self, performance):
        """
        Dynamically plot progress of training.
        """
        plt.figure(figsize=(10,8))
        plt.plot(performance)
        plt.title(f'Training {self.agent}..')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.close()
        plt.pause(0.2)


    def optimize_model(self):
        """
        Perform a single step of the optimization.
        """
        # Check if there are enough samples in the replay memory
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Sample a batch of transitions from the replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)

        # Transpose the batch of transitions - converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask for non-final states and concatenate batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                               batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat([s.to(self.device) for s in batch.state])
        action_batch = torch.cat([a.to(self.device) for a in batch.action])
        reward_batch = torch.cat([r.to(self.device) for r in batch.reward])


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.inference_mode():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss - smooth L1 loss function. 
        # It measures the difference between the predicted Q values and the expected Q values
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        # sets the gradients to zero, backpropagates the loss, applies In-place gradient clipping
        # and updates the model's parameters using the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def dqn_train(self, reward_weight=None, plot_save_path=None, dynamic_plot=False):
        """
        Train the agent using Double DQN.
        """
        if not dynamic_plot:
            progress_bar = tqdm(total=self.num_episodes, desc="Training..")
        
        for i_episode in range(self.num_episodes):

            # Initialize the environment and get its state
            state = torch.from_numpy(self.env.reset()).unsqueeze(0).to(self.device, dtype=torch.float)

            for t in count():
                action = self.select_action(state)
                observation, reward, done, _ = self.env.step(action.item())

                next_state = torch.from_numpy(observation).unsqueeze(0).to(self.device, dtype=torch.float)

                if reward_weight is None:
                    reward = torch.tensor([reward], device=self.device)
                    x, x_dot, theta, theta_dot = observation

                    # calculate the normalized distance of the cart's center from the edges of the screen
                    # penalize the agent for being close to the edges, encouraging it to stay away
                    r1 = (self.env.unwrapped.x_threshold - abs(x)) / self.env.unwrapped.x_threshold - 0.8

                    # calculate the normalized remaining angular range before the pole reaches its maximum angle
                    # penalize the agent for having the pole at a large angle, to keep the pole more upright
                    r2 = (self.env.unwrapped.theta_threshold_radians - abs(theta)) / \
                          self.env.unwrapped.theta_threshold_radians - 0.5

                    reward = torch.tensor([r1 + r2])

                else:
                    # use rewards calculated using weights (IRL)
                    features = self.construct_feature_vector(observation)
                    reward = reward_weight.t() @ features                 # w^T ⋅ φ

                # terminate
                if done:
                    next_state = None

                # store the transition in self.memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # perform one step of the optimization (on the target network)
                self.optimize_model()

                # break if episode is done or exceeds a maximum number of steps
                if done or t > 200:
                    self.episode_durations.append(t + 1)
                    performance = torch.tensor(self.episode_durations, dtype=torch.float).numpy()
                    if dynamic_plot:
                        self.plot_progress(performance)
                    break

            # test model (after at least 100 episodes)
            policy_reward = 0
            if i_episode > self.num_episodes//2:
                policy_reward = self.test_model(self.policy_net)

            # update the target network weights every TARGET_UPDATE episodes
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if not dynamic_plot:
                progress_bar.update(1)

        # Done training
        if dynamic_plot:
            display.clear_output()
        else:
            progress_bar.close()
        self.is_trained = True
        plt.figure(figsize=(10,8))
        plt.plot(performance)
        plt.title(f'Performance of {self.agent}')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        if plot_save_path:
            plt.savefig(plot_save_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
        self.env.close()


    def construct_feature_vector(self, state):
        """
        Normalize state components and construct a feature vector for given state.
        """
        # Normalize state components - horizontal position, velocity, angle of the pole, angular velocity
        x, x_dot, theta, theta_dot = state
        x = (x + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
        x_dot = (x_dot + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
        theta = (theta + self.env.unwrapped.theta_threshold_radians) / \
                (2 * self.env.unwrapped.theta_threshold_radians)
        theta_dot = (theta_dot + self.env.unwrapped.theta_threshold_radians) / \
                (2 * self.env.unwrapped.theta_threshold_radians)

        # Construct feature vector
        feature_vector = torch.tensor([
            x, x_dot, theta, theta_dot,
            x ** 2, x_dot ** 2, theta ** 2, theta_dot ** 2,
        ], dtype=torch.float)

        return feature_vector


    def test_model(self, model, save_states=False, render_save_path=None):
        """
        Run the environment using trained model.
        """
        episode_reward = 0
        state_list = []     # List to store state feature vectors

        observation = self.env.reset()
        state = torch.from_numpy(observation).unsqueeze(0).to(self.device, dtype=torch.float)

        if save_states:
            state_list.append(self.construct_feature_vector(observation))

        images = []

        with torch.inference_mode():
            for t in count():
                if render_save_path:
                    img = self.env.render(mode='rgb_array')
                    images.append(img)
                action = model(state).max(1)[1].view(1, 1)

                observation, reward, done, _ = self.env.step(action.item())

                state = torch.from_numpy(observation).unsqueeze(0).to(self.device, dtype=torch.float)

                if save_states:
                    state_list.append(self.construct_feature_vector(observation))

                episode_reward += reward

                if done or t > 200:
                    break

        # Based on the total reward for the episode, determine the best model
        if episode_reward > self.best_reward and not save_states:
            self.best_reward = episode_reward
            self.best_model = copy.deepcopy(model)

        if render_save_path:
            imageio.mimsave(f'{render_save_path}.gif', images, fps=30, loop=0)
            self.env.close()
            with open(f'{render_save_path}.gif', 'rb') as f:
                display.display(display.Image(data=f.read(), format='gif'))

        if not save_states:
            return episode_reward
        else:
            return episode_reward, state_list


    def save_trained_model(self, save_path=None):
        """
        Save the trained policy network model.
        """
        if not save_path:
            if not os.path.exists('./data'):
                os.makedirs('./data')

            save_path = f'./data/{self.agent}.pt'
            
        if self.is_trained:
            torch.save(self.best_model.state_dict(), save_path)
            print('Saved ',save_path)
        else:
            print('\nModel not trained.')