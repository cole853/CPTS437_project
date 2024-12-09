# a2c_trainer.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
from a2c_agent import Agent
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo

class A2CTrainer:
    def __init__(self, gym_id, total_timesteps, learning_rate=2.5e-4, seed=1):
        self.gym_id = gym_id
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.seed = seed

        # Training hyperparameters
        self.num_steps = 128
        self.batch_size = self.num_steps
        self.update_epochs = 4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.num_minibatches = 4
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        # Create environment to get observation and action spaces
        env = gym.make(self.gym_id)
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n
        env.close()

        # Initialize agent
        self.agent = Agent(env)

        # Optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # Storage for training data
        self.obs = torch.zeros((self.num_steps,) + self.obs_dim)
        self.actions = torch.zeros((self.num_steps,), dtype=torch.long)
        self.logprobs = torch.zeros(self.num_steps)
        self.rewards = torch.zeros(self.num_steps)
        self.dones = torch.zeros(self.num_steps)
        self.values = torch.zeros(self.num_steps)

        # Metrics
        self.ep_rewards = []
        self.steps_per_ep = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_values = []
        self.lrs = []

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir="tensorboard_logs")

    def train(self):
        # Create environment
        env = gym.make(self.gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Initialize environment
        obs, info = env.reset(seed=self.seed)
        next_obs = torch.tensor(obs, dtype=torch.float32)
        next_done = torch.zeros(1)

        num_updates = self.total_timesteps // self.batch_size

        start_time = time.time()
        global_step = 0

        for update in range(1, num_updates + 1):
            # Collect trajectories
            for step in range(self.num_steps):
                global_step += 1
                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, entropy, value = self.agent.get_action_and_value(next_obs)
                self.actions[step] = action
                self.logprobs[step] = logprob
                self.values[step] = value.flatten()

                # Take action in environment
                action_np = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action_np)
                self.rewards[step] = torch.tensor(reward, dtype=torch.float32)
                next_obs = torch.tensor(obs, dtype=torch.float32)
                next_done = torch.tensor(terminated or truncated, dtype=torch.float32)

                # Handle end of episode
                if terminated or truncated:
                    ep_reward = info['episode']['r']
                    ep_steps = info['episode']['l']
                    self.ep_rewards.append(ep_reward)
                    self.steps_per_ep.append(ep_steps)
                    self.writer.add_scalar("Train/Episode Reward", ep_reward, global_step)
                    self.writer.add_scalar("Train/Episode Steps", ep_steps, global_step)
                    print(f"Update {update}, Step {step}, Episode Reward: {ep_reward}, Steps: {ep_steps}")

                    obs, info = env.reset()

            # Compute advantages and returns
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).flatten()
                advantages, returns = self.compute_gae(next_value)

            # Flatten the batch
            b_obs = self.obs.reshape((-1,) + self.obs_dim)
            b_actions = self.actions.reshape(-1)
            b_logprobs = self.logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Optimize policy and value networks
            for epoch in range(self.update_epochs):
                # Shuffle the indices
                indices = np.arange(self.batch_size)
                np.random.shuffle(indices)

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = indices[start:end]

                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logprobs = b_logprobs[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]

                    # Get current policy outputs
                    action_pred, logprob, entropy, value = self.agent.get_action_and_value(mb_obs, mb_actions)

                    # Calculate ratio (for PPO), but A2C does not use clipping, so we will not use it in a clipped way
                    ratio = torch.exp(logprob - mb_logprobs)

                    # Policy loss (A2C style: no clipping)
                    pg_loss = (-mb_advantages * ratio).mean()

                    # Value loss (A2C style: just MSE, no clipping)
                    newvalue = value.flatten()
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # Record losses
                    self.value_losses.append(v_loss.item())
                    self.policy_losses.append(pg_loss.item())
                    self.entropy_values.append(entropy_loss.item())

            # Logging
            if update % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Update {update}/{num_updates}, Time Elapsed: {elapsed_time:.2f}s")
                self.writer.add_scalar("Train/Value Loss", np.mean(self.value_losses[-self.num_steps:]), update)
                self.writer.add_scalar("Train/Policy Loss", np.mean(self.policy_losses[-self.num_steps:]), update)
                self.writer.add_scalar("Train/Entropy", np.mean(self.entropy_values[-self.num_steps:]), update)

        # Save the trained model
        os.makedirs("agents", exist_ok=True)
        model_path = os.path.join("agents", f"a2c_{self.gym_id}_{int(time.time())}.pth")
        torch.save(self.agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Plot training metrics
        self.plot_metrics()

        # Close environment and TensorBoard writer
        env.close()
        self.writer.close()

    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = torch.zeros(self.num_steps)
        returns = torch.zeros(self.num_steps)
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + self.gamma * (1 - self.dones[step]) * next_value - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages[step] = gae
            returns[step] = gae + self.values[step]
            next_value = self.values[step]
        return advantages, returns

    def plot_metrics(self):
        """
        Plot training metrics and save them.
        """
        os.makedirs("plots", exist_ok=True)
        run_name = f"a2c_{self.gym_id}_{int(time.time())}"
        plot_dir = os.path.join("plots", run_name)
        os.makedirs(plot_dir, exist_ok=True)

        # Episode Rewards
        smoothed_rewards = pd.Series(self.ep_rewards).rolling(window=50, min_periods=1).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(self.ep_rewards, label="Episode Rewards")
        plt.plot(smoothed_rewards, label="Smoothed Rewards (50)")
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "ep_rewards.png"))
        plt.close()

        # Policy Loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.policy_losses, label="Policy Loss")
        plt.title("Policy Loss Over Time")
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "policy_loss.png"))
        plt.close()

        # Value Loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.value_losses, label="Value Loss")
        plt.title("Value Loss Over Time")
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "value_loss.png"))
        plt.close()

        # Entropy
        plt.figure(figsize=(10, 6))
        plt.plot(self.entropy_values, label="Entropy")
        plt.title("Entropy Over Time")
        plt.xlabel("Updates")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "entropy.png"))
        plt.close()

        print(f"Plots saved in {plot_dir}")

if __name__ == "__main__":
    trainer = A2CTrainer(gym_id='LunarLander-v3', total_timesteps=10000000)
    trainer.train()
