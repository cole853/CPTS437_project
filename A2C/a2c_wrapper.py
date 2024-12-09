import os
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import RecordVideo
from a2c_trainer import A2CTrainer
from a2c_agent import Agent

class a2c_Wrapper:
    def __init__(self):
        pass

    def playGame(self):
        """
        Play a random game (or baseline policy game) to confirm the environment runs.
        """
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        obs, info = env.reset(seed=1)

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                done = True

        env.close()
        return total_reward, steps

    def trainModel_A2C(self, total_timesteps):
        """
        Train the A2C model using A2CTrainer for a specified number of timesteps.
        The model, logs, and plots will be saved automatically.
        """
        trainer = A2CTrainer(gym_id='LunarLander-v3', total_timesteps=total_timesteps)
        trainer.train()
        print("A2C model training completed. Check the 'agents' folder for the saved model.")

    def testModel_A2C(self, model_path):
        """
        Test a trained A2C model on the LunarLander-v3 environment.
        Loads the model and runs one full episode, returning the score and steps.
        """
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        obs, info = env.reset(seed=1)

        agent = Agent(env)
        agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.eval()

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                done = True

        env.close()
        return total_reward, steps

    def load_model_A2C(self, model_path):
        """
        Load a trained A2C model from the given path and return the agent.
        """
        # Create a temporary environment to load dimensions
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        agent = Agent(env)
        agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.eval()
        env.close()
        return agent

    def run_single_episode_A2C(self, model_path):
        """
        Run a single episode with a loaded A2C model and print the total reward.
        """
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        agent = self.load_model_A2C(model_path)

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            done = terminated or truncated

        env.close()
        print(f"Total Reward for the Episode: {total_reward:.2f}")

    def evaluate_A2C(self, model_path, num_episodes=20, video_folder="videos"):
        """
        Evaluate the A2C agent for multiple episodes and record videos.
        Returns a list of episode rewards.
        """
        os.makedirs(video_folder, exist_ok=True)
        base_env = gym.make('LunarLander-v3', render_mode='rgb_array')
        env = RecordVideo(base_env, video_folder=video_folder, video_length=1000, name_prefix="a2c_evaluation")

        agent = self.load_model_A2C(model_path)

        episode_rewards = []
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action_np = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action_np)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward}")
        
        env.close()
        return episode_rewards

    def plot_evaluation(self, rewards, save_path="plots/evaluation_rewards.png"):
        """
        Plot the evaluation rewards and mark the last 20 episodes.
        """
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, marker='o', label="All Episodes")
        if len(rewards) >= 20:
            plt.axvline(len(rewards) - 20, color="r", linestyle="--", label="Start of Last 20 Episodes")
        plt.title("Evaluation Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.close()

    def summary_evaluation(self, rewards):
        """
        Print a summary of the last 20 episodes.
        """
        if len(rewards) < 20:
            print("Not enough episodes to compute a 20-episode summary.")
            return

        last_20 = rewards[-20:]
        average_reward = np.mean(last_20)
        std_reward = np.std(last_20)

        print("\n--- Evaluation Summary (Last 20 Episodes) ---")
        print(f"Average Reward (Last 20 Episodes): {average_reward:.2f}")
        print(f"Standard Deviation (Last 20 Episodes): {std_reward:.2f}")
