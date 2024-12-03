# evaluate.py

import os
import torch
import gymnasium as gym
from a2c_agent import Agent
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

def load_model(model_path, env):
    agent = Agent(env)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()
    return agent

def evaluate(agent, env, num_episodes=5, video_folder="videos"):
    """
    Evaluate the agent and record videos.
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
    model_path = "agents/a2c_LunarLander-v3_1733200628.pth"  # Replace with your actual model path

    # Create environment
    env = gym.make('LunarLander-v3', render_mode='rgb_array')

    # Load the trained agent
    agent = load_model(model_path, env)

    # Evaluate the agent
    rewards = evaluate(agent, env)

    # Plot evaluation rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='o')
    plt.title("Evaluation Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("plots/evaluation_rewards.png")
    plt.show()

if __name__ == "__main__":
    main()