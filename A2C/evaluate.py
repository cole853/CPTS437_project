import os
import torch
import gymnasium as gym
from a2c_agent import Agent
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np


def load_model(model_path, env):
    agent = Agent(env)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()
    return agent

def run_single_episode(agent, env):
    """
    Run the agent on the environment for one episode and print the reward.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        action_np = action.cpu().numpy()  # Convert action to numpy for the environment
        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        total_reward += reward

    print(f"Total Reward for the Episode: {total_reward:.2f}")

def evaluate(agent, env, num_episodes=20, video_folder="videos"):
    """
    Evaluate the agent and record videos for all episodes.
    """
    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(env, video_folder=video_folder, video_length=1000, name_prefix="a2c_evaluation")

    episode_rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    env.close()
    return episode_rewards


def main():
    # Path to the trained model
    model_path = "agents/a2c_LunarLander-v3_1733696509.pth"  # Replace with your actual model path

    # Create environment
    env = gym.make('LunarLander-v3', render_mode='rgb_array')

    # Load the trained agent
    agent = load_model(model_path, env)

    run_single_episode(agent, env)

    # Evaluate the agent for a total number of episodes
    total_episodes = 50  # Total number of episodes to evaluate
    rewards = evaluate(agent, env, num_episodes=total_episodes)

    # Extract and calculate the statistics for the last 20 episodes
    last_20_rewards = rewards[-20:]  # Select the last 20 rewards
    average_reward = np.mean(last_20_rewards)
    std_reward = np.std(last_20_rewards)

    print("\n--- Evaluation Summary (Last 20 Episodes) ---")
    print(f"Average Reward (Last 20 Episodes): {average_reward:.2f}")
    print(f"Standard Deviation (Last 20 Episodes): {std_reward:.2f}")

    # Plot evaluation rewards
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='o', label="All Episodes")
    plt.axvline(len(rewards) - 20, color="r", linestyle="--", label="Start of Last 20 Episodes")
    plt.title("Evaluation Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig("plots/evaluation_rewards.png")
    plt.show()


if __name__ == "__main__":
    main()
