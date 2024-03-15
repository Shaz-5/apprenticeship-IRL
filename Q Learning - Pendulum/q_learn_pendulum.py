import gym
import numpy as np
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, Image

# Train agent using Q Learning
class QTrainer:
    
    def __init__(self, env, discrete_space_size=[21, 21, 65], discrete_act_size=17, GAMMA=0.95, ALPHA=0.1):
        self.env = env
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.discrete_space_size = discrete_space_size
        self.discrete_act_size = discrete_act_size
        self.action_space = self.discretize_action_space()
        self.Q = self.init_Q()
        self.episode_lengths = []
        self.episode_rewards = []
        self.total_rewards = []
        
        
    def discretize_state(self, state):
        """
        Discretize the continuous observable state space.
        """
        # calculate discrete bin size
        bin_size = (self.env.observation_space.high - self.env.observation_space.low) / [i-1 for i in self.discrete_space_size]
        # normalize and divide by bin size
        discrete_state = tuple(((state - self.env.observation_space.low) / bin_size).astype(np.int32))
        return discrete_state
    
    
    def discretize_action_space(self):
        """
        Discretize the continuous action space.
        """
        action_space = {}
        action_bin_size = (self.env.action_space.high - self.env.action_space.low) / (self.discrete_act_size - 1)
        for i in range(self.discrete_act_size):
            action_space[i] = [self.env.action_space.low[0] + (i * action_bin_size[0])]
        return action_space
    
    
    def init_Q(self, low=-2, high=0):
        """
        Initialize Q Table
        """
        Q = np.random.uniform(low=low, high=high, size=(self.discrete_space_size + [self.discrete_act_size]))
        return Q
    
    
    def sigmoid(self, array):
        return 1 / (1 + np.exp(-array))
    
    
    def q_learning_train(self, num_episodes=10000, irl=False, weight=None, 
                         print_interval=2000, save_q_path=None, render_save_path=None):
        """
        Train a Q-learning agent through multiple episodes.

        Parameters:
        - num_episodes: Number of training episodes.
        - irl: if True, trains using obtained reward
        - weight: Weight obtained from the IRL algorithm.
        - print_interval(Optional): Interval for printing training progress.
        - save_q_path: Path to save the trained Q Table progressively.
        - render_save_path: Path to save render of training progress (every 10000 episodes)

        Returns:
        - episode_lengths: List of lengths for each episode.
        - episode_rewards: List of rewards for each episode.
        - Q: Trained Q-table.
        """
        if render_save_path:
            self.env = gym.wrappers.Monitor(self.env, render_save_path, 
                                            video_callable=lambda x: x % (num_episodes//10) == 0, force=True)
        Q = self.init_Q()
        episode_lengths = []
        episode_rewards = []
        epsilon = 1
        epsilon_final = 0.1
        epsilon_decay = 0.999

        for episode in tqdm(range(1, num_episodes + 1), desc="Training Episodes.."):

            observation = self.env.reset()
            done = False
            move_count = 0   # no. of moves in an episode
            state = self.discretize_state(observation)
            total_reward = 0

            while not done:
                move_count += 1

                if np.random.uniform() < epsilon:
                    action = np.random.randint(0, self.discrete_act_size)  # epsilon-greedy exploration
                else:
                    action = np.argmax(Q[state])   # action with max value

                torque = self.action_space[action]
                observation, reward, done, _ = self.env.step(torque)
                
                new_state = self.discretize_state(observation)

                if irl:
                    observation = -1 * self.sigmoid(observation)

                    # discard the simulation reward and use the reward function found from the IRL algorithm
                    reward = np.dot(weight, observation)     # wT · φ

                total_reward += reward

                neg_reward = -1 if irl else -300
                # penalize early episode termination
                if done and move_count < 200:
                    reward = neg_reward

                max_q_s1a1 = np.max(Q[new_state])
                current_q = Q[state + (action,)]
                new_q = (1 - self.ALPHA) * current_q + self.ALPHA * (reward + self.GAMMA * max_q_s1a1)
                
                Q[state + (action,)] = new_q
                
                if new_state[0] == 0 and new_state[1] == 1:
                    Q[state + (action,)] = 0
                    print(f"\nGoal acheived in {episode}.")
                
                state = new_state

            if epsilon >= epsilon_final:
                epsilon *= epsilon_decay
                
            episode_reward, episode_length = total_reward, move_count

            if save_q_path and episode % 10 == 0:
                np.save(save_q_path, Q)
                
            if episode % print_interval == 0:
                print(f"Episode: {episode}, Epsilon: {epsilon:.4f}, Reward: {episode_reward}")

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            
        if not irl:
            self.Q = Q

        if irl:
            avg_length = np.average(episode_lengths)
            std_dev_length = np.std(episode_lengths)

            print(f"Avg Length: {avg_length} \nStandard Deviation: {std_dev_length}")

        self.env.close()
        
        return episode_lengths, episode_rewards, Q
        
    
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
        if Q is None:
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

                state = self.discretize_state(observation)
                action = np.argmax(Q[state])
                torque = self.action_space[action]
                observation, reward, done, _ = self.env.step(torque)
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
            
        fig = plt.figure(figsize=(8,6))
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