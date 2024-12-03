import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Save training data
def save_training_data(model, optimizer, replay_buffer, episode, filepath=os.path.join(base_dir, "agents", "dqn_checkpoint.pth")):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'replay_buffer': list(replay_buffer),
        'episode': episode
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    print("Checkpoint saved successfully!")

# Load training data
def load_training_data(model, optimizer, filepath="agents/dqn_checkpoint.pth"):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        replay_buffer = deque(checkpoint['replay_buffer'], maxlen=50000)
        start_episode = checkpoint['episode']
        print("Checkpoint loaded successfully!")
        return replay_buffer, start_episode
    else:
        print("No checkpoint found. Starting fresh training.")
        return deque(maxlen=50000), 0

# Save plots
def save_plots(ep_rewards, steps_per_ep, run_name):
    plot_dir = os.path.join(base_dir, "plots", run_name)
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

# Function to write episode data to CSV
def log_episode_data(run_name, episode, total_reward, steps, epsilon):
    log_dir = os.path.join(base_dir, "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.csv")
    
    # Check if the file exists; if not, write the header
    if not os.path.exists(log_file):
        with open(log_file, mode='w') as f:
            f.write("Episode,TotalReward,Steps,Epsilon\n")
    
    # Append the episode data
    with open(log_file, mode='a') as f:
        f.write(f"{episode},{total_reward},{steps},{epsilon}\n")
    print(f"Logged Episode {episode} data to {log_file}")

# Initialize environment and hyperparameters
run_name = "dqn_lunar_lander"
os.makedirs(os.path.join(base_dir, "videos", run_name), exist_ok=True)
env = gym.make(
    'LunarLander-v3',
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode='rgb_array'
)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.RecordVideo(env, os.path.join(base_dir, "videos", run_name))

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Load existing training data if available
replay_buffer, start_episode = load_training_data(q_net, optimizer)

gamma = 0.99
batch_size = 128
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
target_update_freq = 100
num_episodes = 10000
ep_rewards = []
steps_per_ep = []

# Training loop
for episode in range(start_episode, num_episodes):
    state = env.reset()[0]
    total_reward = 0
    steps = 0
    
    for t in range(1000):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = q_net(state_tensor).argmax().item()
        
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
        
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)
            
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            loss = loss_fn(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    ep_rewards.append(total_reward)
    steps_per_ep.append(steps)
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Log episode data to CSV
    log_episode_data(run_name, episode + 1, total_reward, steps, epsilon)
    
    if (episode + 1) % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())
    
    # Save progress every 50 episodes
    if (episode + 1) % 50 == 0:
        save_training_data(q_net, optimizer, replay_buffer, episode + 1)

# Save final training data and plots
save_training_data(q_net, optimizer, replay_buffer, num_episodes)
save_plots(ep_rewards, steps_per_ep, run_name)
print("Training complete. Videos, plots, and logs saved!")