import pygame
import gymnasium as gym
from PPO_trainer import PPO_trainer
from Q_agent import Q_agent
import time
import os
from torch.utils.tensorboard import SummaryWriter


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

        env.close()

    # trains the PPO model on the environment
    # steps is the number of steps to train the agent on
    def trainModel_PPO(self, steps):
        trainer = PPO_trainer(self.gym_id, steps)
        trainer.run()

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

        exp_name = os.path.basename(__file__).rstrip(".py")
        seed = 1
        run_name = f"{self.gym_id}__{exp_name}__{seed}__{int(time.time())}"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)

        # make summary writer
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % "\n",
        )

        total_steps = 0
        while total_steps < steps:
            obs, info = env.reset()
            done = False
            episode_return = 0
            ep_steps = 0

            obs = tuple([round(x) for x in obs])

            # play one episode
            while not done and ep_steps < 128:
                action = 0
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

                writer.add_scalar("charts/epsilon", epsilon, total_steps)
                writer.add_scalar("losses/error", error, total_steps)

            print(f"Total Steps {total_steps} | Episode Return {episode_return}")
            writer.add_scalar("charts/episodic_return", episode_return, total_steps)

        agent.save_policy(f"runs/{run_name}")
        env.close()
