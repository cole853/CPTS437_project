# This Wrapper file is uses the functionallity established in dqn.py in a wrapper format.
# This allows it to be used in the collab notebook.

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pandas as pd
import matplotlib.pyplot as plt

#Initialize neural network object
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), # Input layer
            nn.ReLU(), # ReLU activation
            nn.Linear(128, 128), # Two hidden layers with 128 nuerons each
            nn.ReLU(), # ReLU activation
            nn.Linear(128, output_dim) # Output layer
        )
    
    def forward(self, x):
        return self.fc(x)

# Main wrapper class
class Wrapper:
    def __init__(self, env_name='LunarLander-v3', run_name="dqn_agent", base_dir="."):
        self.env_name = env_name
        self.run_name = run_name
        self.base_dir = base_dir
        
        # Initialize environment
        self.env = gym.make(
            env_name,
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
            render_mode='rgb_array'
        )
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        video_dir = os.path.join(self.base_dir, "videos", run_name)
        os.makedirs(video_dir, exist_ok=True)
        self.env = gym.wrappers.RecordVideo(self.env, video_dir)
        
        # Define model and parameters
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=50000)
        self.gamma = 0.99
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.target_update_freq = 100
    

    # Train model function. Ends 
    def trainModel(self, max_steps=10000):
        ep_rewards = []
        steps_per_ep = []
        actual_episodes = 0  # Track actual episodes run
        total_steps = 0  # Track total training steps globally

        while total_steps < max_steps:
            state = self.env.reset()[0]
            total_reward = 0
            steps = 0
            actual_episodes += 1  # Increment actual episode count

            for t in range(1000):  # Max steps per episode
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = self.q_net(state_tensor).argmax().item()

                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                steps += 1
                total_steps += 1  # Increment global step count

                if done or total_steps >= max_steps:
                    break

                if len(self.replay_buffer) >= self.batch_size:
                    batch = random.sample(self.replay_buffer, self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.FloatTensor(np.array(states))
                    actions = torch.LongTensor(actions)
                    rewards = torch.FloatTensor(rewards)
                    next_states = torch.FloatTensor(np.array(next_states))
                    dones = torch.FloatTensor(dones)

                    q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                    next_q_values = self.target_net(next_states).max(1)[0]
                    target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

                    loss = self.loss_fn(q_values, target_q_values.detach())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            ep_rewards.append(total_reward)
            steps_per_ep.append(steps)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.log_episode_data(actual_episodes, total_reward, steps, self.epsilon)

            if actual_episodes % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Check if total steps exceed the maximum allowed steps
            if total_steps >= max_steps:
                print(f"Reached maximum training steps: {total_steps}. Stopping training.")
                break

        self.save_plots(ep_rewards, steps_per_ep)
        print(f"Training complete. Total episodes run: {actual_episodes}, Total steps: {total_steps}")
        self.save_model()



    def save_model(self):
        agent_dir = os.path.join(self.base_dir, "DQN", "agents")  # Corrected to "agents" 
        os.makedirs(agent_dir, exist_ok=True)
        model_path = os.path.join(agent_dir, "dqn_model.pth")
        optimizer_path = os.path.join(agent_dir, "optimizer.pth")
        torch.save(self.q_net.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        print(f"Model saved to {model_path}")
        print(f"Optimizer saved to {optimizer_path}")

    def testModel(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
        state = self.env.reset()[0]
        total_reward = 0
        steps = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.q_net(state_tensor).argmax().item()
            next_state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                break
        
        print(f"Test completed: Total reward = {total_reward}, Steps = {steps}")
        return total_reward, steps

    def save_plots(self, ep_rewards, steps_per_ep):
        plot_dir = os.path.join(self.base_dir, "plots", self.run_name)
        os.makedirs(plot_dir, exist_ok=True)

        # Plot for Episode Rewards
        smoothed_rewards = pd.Series(ep_rewards).rolling(window=100, center=True).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(ep_rewards, label="Episode Rewards", marker="o")
        plt.plot(smoothed_rewards, label="Smoothed Rewards (Moving Average 100)", linestyle="--")
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "ep_rewards.png"), dpi=300)
        plt.clf()

        # Plot for Steps per Episode
        smoothed_steps = pd.Series(steps_per_ep).rolling(window=100, center=True).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(steps_per_ep, label="Steps per Episode", marker="o")
        plt.plot(smoothed_steps, label="Smoothed Steps per Episode (Moving Average 100)", linestyle="--")
        plt.title("Steps per Episode Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "steps_per_ep.png"), dpi=300)
        plt.clf()

        print(f"Plots saved in the directory: {plot_dir}")

    def log_episode_data(self, episode, total_reward, steps, epsilon):
        log_dir = os.path.join(self.base_dir, "logs", self.run_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training_log.csv")
        
        # Check if the file exists; if not, write the header
        if not os.path.exists(log_file):
            with open(log_file, mode='w') as f:
                f.write("Episode,TotalReward,Steps,Epsilon\n")
        
        # Append the episode data
        with open(log_file, mode='a') as f:
            f.write(f"{episode},{total_reward},{steps},{epsilon}\n")
        print(f"Episode {episode} -- Total Steps: {steps} -- Reward: {total_reward}")