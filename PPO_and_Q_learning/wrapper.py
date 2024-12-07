import pygame
import gymnasium as gym
from PPO_trainer import PPO_trainer
from Q_agent import Q_agent
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from PPO_agent import Agent
import torch


# a wrapper class for the game
# includes functions for training the Q learning and PPO algorithms
class Wrapper:
    def __init__(self):
        self.gym_id = "LunarLander-v3"

    # allows the user to play the game without the agent
    # the up arrow is gas
    # the down arrow is brake
    # the left arrow turns left
    # the right arrow turns right
    def playGame(self):
        env = gym.make(self.gym_id, continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human')
        pygame.init()
        observation, info = env.reset()

        ep_reward = 0
        num_steps = 0
        episode_over = False
        while not episode_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    episode_over = True

            action = 0

            keys = pygame.key.get_pressed()  # Get the state of all keyboard keys

            # Map keys to actions
            if keys[pygame.K_UP]:  # main engine
                action = 2
            elif keys[pygame.K_LEFT]:  # Steer left
                action = 1
            elif keys[pygame.K_RIGHT]:  # Steer right
                action = 3

            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated
            ep_reward += reward
            num_steps += 1

        env.close()

        return ep_reward, num_steps

    # trains the PPO model on the environment
    # steps is the number of steps to train the agent on
    def trainModel_PPO(self, steps):
        trainer = PPO_trainer(self.gym_id, steps)
        trainer.run()
        sys.exit(0)

    # runs the trained PPO model on one simulation
    # returns the reward and steps
    # takes in the path to load the model from
    def testModel_PPO(self, path):
        env = gym.make(self.gym_id, continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human')
        pygame.init()
        obs, info = env.reset()

        model = Agent(env)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()

        ep_reward = 0
        num_steps = 0
        episode_over = False
        while not episode_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    episode_over = True

            obs = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs.clone().detach()

            action, logprob, _, value = model.get_action_and_value(obs)

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action.item())

            episode_over = terminated
            ep_reward += reward
            num_steps += 1

        env.close()

        return ep_reward, num_steps

    # trains the Q learning model to play the game
    def trainModel_Q(self, steps, learning_rate=2.5e-4, start_epsilon=1.0, final_epsilon=0.1):
        epsilon_decay = start_epsilon / (steps / 2)  # reduce the exploration over time

        env = gym.make(self.gym_id, continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='rgb_array')

        agent = Q_agent(
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon
        )

        exp_name = "Q_learning"
        seed = 1
        run_name = f"{self.gym_id}__{exp_name}__{seed}__{int(time.time())}"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)

        total_steps = 0
        num_episodes = 0
        ep_rewards = []
        steps_per_ep = []
        losses = []
        epsilons = []
        while total_steps < steps:
            num_episodes += 1
            obs, info = env.reset()
            done = False
            episode_return = 0
            ep_steps = 0

            obs = tuple([round(x) for x in obs])

            # play one episode
            while not done and ep_steps < 300:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                next_obs = tuple([round(x) for x in next_obs])
                episode_return += reward

                # update the agent
                error = agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

                total_steps += 1
                ep_steps += 1
                epsilon = agent.decay_epsilon()

                if total_steps % 100 == 0:
                    losses.append(error)
                    epsilons.append(epsilon)

            ep_rewards.append(episode_return)
            steps_per_ep.append(ep_steps)
            print(f"Total Steps {total_steps} | Episode Return {episode_return}")

        os.makedirs("agents", exist_ok=True)
        agent_dir = os.path.join("agents", run_name)

        agent.save_policy(agent_dir)
        self.save_plots(ep_rewards, steps_per_ep, losses, epsilons, run_name)

        env.close()

        sys.exit(0)

    # runs the trained Q model on one simulation
    # returns the reward and steps
    # takes in the path to load the model from
    def testModel_Q(self, path):
        env = gym.make(self.gym_id, continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human')
        pygame.init()
        obs, info = env.reset()

        learning_rate = 2.5e-4
        start_epsilon = 0
        final_epsilon = 0
        epsilon_decay = 0

        agent = Q_agent(
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon
        )

        agent.load_policy(path)

        ep_reward = 0
        num_steps = 0
        episode_over = False
        while not episode_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    episode_over = True

            obs = tuple([round(x) for x in obs])
            action = agent.get_action(obs)

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated
            ep_reward += reward
            num_steps += 1

        env.close()

        return ep_reward, num_steps

    # saves plots as in plots/run_name
    def save_plots(self, ep_rewards, steps_per_ep, losses, epsilons, run_name):
        # make directory to store plots
        os.makedirs("plots", exist_ok=True)
        plot_dir = os.path.join("plots", run_name)
        os.makedirs(plot_dir, exist_ok=True)

        # Plot for Episode Rewards
        smoothed_rewards = pd.Series(ep_rewards).rolling(window=500, center=True).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(ep_rewards, label="Episode Rewards", marker="o")
        plt.plot(smoothed_rewards, label="Smoothed rewards (Moving Average 500)", linestyle="--")
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "ep_rewards.png"), dpi=300)
        plt.clf()

        # Plot for Steps per Episode
        smoothed_steps = pd.Series(steps_per_ep).rolling(window=500, center=True).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(steps_per_ep, label="Steps per Episode", marker="o")
        plt.plot(smoothed_steps, label="Smoothed Steps per Episode (Moving Average 500)", linestyle="--")
        plt.title("Steps per Episode Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "steps_per_ep.png"), dpi=300)
        plt.clf()

        # Plot for Losses
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Losses", marker="o")
        plt.title("Losses Over Time")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "losses.png"), dpi=300)
        plt.clf()

        # Plot for Epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, label="Epsilon", marker="o")
        plt.title("Epsilon Over Time")
        plt.xlabel("Step")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_dir, "epsilons.png"), dpi=300)
        plt.clf()

        print(f"Plots saved in the directory: {plot_dir}")
