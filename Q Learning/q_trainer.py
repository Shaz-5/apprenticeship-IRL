import gym
import numpy as np
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, Image

class QTrainer:
    
    def __init__(self, env, nbins=10, GAMMA=0.9, ALPHA=0.01):
        self.env = env
        self.nbins = nbins
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.bins = self.discretize_state_space()
        self.states = self.get_states()
        self.Q = self.init_Q()
        self.episode_lengths = []
        self.episode_rewards = []
        self.total_rewards = []
        
    def discretize_state_space(self):
        """
        Discretize the continuous observable state space.
        """
        bins = np.zeros((4, self.nbins))

        bins[0] = np.linspace(-4.8, 4.8, self.nbins)       # position
        bins[1] = np.linspace(-5, 5, self.nbins)           # velocity
        bins[2] = np.linspace(-0.418, 0.418, self.nbins)   # angle
        bins[3] = np.linspace(-5, 5, self.nbins)           # tip velocity

        return bins
    
    
    def get_states(self):
        """
        Generate a list of states based on the number of bins.
        """
        states = []

        for i in range(self.nbins + 1):
            for j in range(self.nbins + 1):
                for k in range(self.nbins + 1):
                    for l in range(self.nbins + 1):
                        state_str = str(i).zfill(2) + str(j).zfill(2) + str(k).zfill(2) + str(l).zfill(2)
                        states.append(state_str)

        # no. of states = (nbins+1)^4
        return states
    
    
    def init_Q(self):
        """
        Initialize Q Table
        """
        Q = {}

        for state in self.states:
            Q[state] = {}

            for action in range(self.env.action_space.n):
                Q[state][action] = 0

        return Q
    
    
    def assign_bins(self, observation):
        """
        Discretize observation into self.bins
        """

        discretized_state = np.zeros(4)

        for i in range(4):
            discretized_state[i] = np.digitize(observation[i], self.bins[i])

        return discretized_state
    
    
    def get_state_as_str(self, state):
        """
        Encode state into string representation
        """

        state_string = ''.join(str(int(e)).zfill(2) for e in state)

        return state_string
    
    
    def get_best_action_value(self, state, Q=None):
        """
        Returns the action with maximum value and the maximum value.
        """
        if not Q:
            Q = self.Q
        return max(Q[state].items(), key=lambda x: x[1])
    
    
    def q_learning_train(self, num_episodes=10000, print_interval=2000):
        """
        Train a Q-learning agent through multiple episodes.

        Parameters:
        - num_episodes: Number of training episodes.
        - print_interval(Optional): Interval for printing training progress.

        Returns:
        - episode_lengths: List of lengths for each episode.
        - episode_rewards: List of rewards for each episode.
        - Q: Trained Q-table.
        """

        episode_lengths = []
        episode_rewards = []

        for episode in tqdm(range(1, num_episodes + 1), desc="Training Episodes.."):
            epsilon = 1.0 / np.sqrt(episode + 1)

            observation = self.env.reset()
            done = False
            move_count = 0   # no. of moves in an episode
            state = self.get_state_as_str(self.assign_bins(observation))
            total_reward = 0

            while not done:
                move_count += 1

                if np.random.uniform() < epsilon:
                    action = self.env.action_space.sample()  # epsilon-greedy exploration
                else:
                    action = self.get_best_action_value(state)[0]   # action with max value

                observation, reward, done, _ = self.env.step(action)

                total_reward += reward

                # penalize early episode termination
                if done and move_count < 200:
                    reward = -300

                new_state = self.get_state_as_str(self.assign_bins(observation))

                best_action, max_q_s1a1 = self.get_best_action_value(new_state)
                self.Q[state][action] += self.ALPHA * (reward + self.GAMMA * max_q_s1a1 - self.Q[state][action])
                state = new_state

            episode_reward, episode_length = total_reward, move_count

            if episode % print_interval == 0:
                print(f"Episode: {episode}, Epsilon: {epsilon:.4f}, Reward: {episode_reward}")

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)

        self.env.close()

        self.episode_lengths = episode_lengths
        self.episode_rewards = episode_rewards
        
        return episode_lengths, episode_rewards, self.Q
    
    
    def run_policy(self, Q=None, num_episodes=1000, render=False, render_filename='Agent Policy'):
        """
        Run the environment using a trained Q Table.

        Parameters:
        - num_episodes: Number of episodes to run the policy.
        - render: If True, render the environment during execution.
        - render_filename: File name (or path) of gif of the render to be saved if rendering.

        Returns:
        - total_rewards: List of total rewards obtained in each episode.
        """
        if not Q:
            Q = self.Q
        total_rewards = []

        for episode in tqdm(range(num_episodes), desc='Running Policy...'):
            observation = self.env.reset()
            done = False
            episode_reward = 0

            images = []
            while not done:
                if render:
                    img = self.env.render(mode='rgb_array')
                    images.append(img)

                state = self.get_state_as_str(self.assign_bins(observation))
                action = self.get_best_action_value(state, Q)[0]
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        if render:
            imageio.mimsave(f'{render_filename}.gif', images, fps=30, loop=0)
            self.env.close()
            with open(f'{render_filename}.gif', 'rb') as f:
                display(Image(data=f.read(), format='gif'))

        self.total_rewards = total_rewards
        return total_rewards
    
    
    def plot_performance(self, total_rewards=None, title='Running Average Reward', save=False, filename='Result'):
        """
        Plot the running average of rewards during training.
        """
        if not total_rewards:
            total_rewards = self.total_rewards
            
        fig = plt.figure()
        num_episodes = len(total_rewards)
        running_avg = np.empty(num_episodes)

        for episode in range(num_episodes):
            running_avg[episode] = np.mean(total_rewards[max(0, episode - 100):(episode + 1)])

        plt.plot(running_avg)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Running Average Reward")

        if save:
            plt.savefig(f"{filename}.png", bbox_inches='tight')
            plt.show()
        else:
            plt.show()
            
            
    def plot_reward_dist(self, rewards=None, title='Reward Distribution'):
        """
        Visualize the reward distribution
        """
        if not rewards:
            rewards = self.total_rewards
            
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # histogram
        axes[0].hist(rewards, bins=20, color='skyblue', edgecolor='black')
        axes[0].set_title('Normal Scale')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')

        # log-scaled histogram (for skewed distributions)
        axes[1].hist(rewards, bins=20, color='skyblue', edgecolor='black', log=True)
        axes[1].set_title('Log Scale')
        axes[1].set_xlabel('Reward')
        axes[1].set_ylabel('Frequency (log scale)')

        plt.suptitle(title)
        plt.show()