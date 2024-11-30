# THIS FILE INCLUDES CODE FROM COSTA HUANG'S PPO TUTORIAL https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
from torch.utils.tensorboard import SummaryWriter
from PPO_agent import Agent


# a wrapper for training the PPO agent
class PPO_trainer:
    def __init__(self, gym_id, steps, learning_rate=2.5e-4):
        self.gym_id = gym_id
        self.exp_name = os.path.basename(__file__).rstrip(".py")
        self.seed = 1
        self.torch_deterministic = True
        self.learning_rate = learning_rate
        self.num_steps = 128
        self.total_timesteps = steps
        self.batch_size = self.num_steps
        self.update_epochs = 4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.num_minibatches = 4
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.norm_adv = True
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.max_grad_norm = 0.5
        self.vf_coef = 0.5
        self.global_step = 0

        # seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # create environment to get observation and action spaces
        env = gym.make(self.gym_id, continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='rgb_array')

        # storage for data collected while running simulations
        self.obs = torch.zeros((self.num_steps,) + env.observation_space.shape)
        self.actions = torch.zeros((self.num_steps,) + env.action_space.shape)
        self.logprobs = torch.zeros(self.num_steps)
        self.rewards = torch.zeros(self.num_steps)
        self.dones = torch.zeros(self.num_steps)
        self.values = torch.zeros(self.num_steps)

        # close environment after getting observation and action spaces
        env.close()

    def run(self):
        # make environment
        run_name = f"{self.gym_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        env = gym.make(self.gym_id, continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=self.seed)

        # make summary writer
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % "\n",
        )

        # make agent and optimizer
        agent = Agent(env)
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # values for the game
        start_time = time.time()
        obs_array, _ = env.reset()
        next_obs = torch.tensor(obs_array, dtype=torch.float32)
        next_done = torch.zeros(1)
        num_updates = self.total_timesteps // self.batch_size

        # loop for the number of episodes run
        for update in range(1, num_updates + 1):
            self.updateLR(update, optimizer)

            # run a simulation and collect the information needed to update the model
            next_obs, next_done = self.simulate_steps(env, agent, writer, next_obs, next_done)

            # compute the advantages to update the model
            advantages, returns = self.compute_gae(agent, next_obs, next_done)

            # flatten the batch
            self.b_obs = self.obs.reshape((-1,) + env.observation_space.shape)
            self.b_logprobs = self.logprobs.reshape(-1)
            self.b_actions = self.actions.reshape((-1,) + env.action_space.shape)
            self.b_advantages = advantages.reshape(-1)
            self.b_returns = returns.reshape(-1)
            self.b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    # optimize the agent based on the minibatch
                    pg_loss, v_loss, entropy_loss = self.optimize_minibatch(agent, optimizer, mb_inds)

            y_pred, y_true = self.b_values.cpu().numpy(), self.b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # record information to show the effect of training
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], self.global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
            print("Samples per second:", int(self.global_step / (time.time() - start_time)))
            writer.add_scalar("charts/samples_per_second", int(self.global_step / (time.time() - start_time)), self.global_step)

        # save model, close environment, and close summarywriter
        torch.save(agent.state_dict(), f"runs/{run_name}")
        env.close()
        writer.close()

    # changes the learning rate to learn less as the agent goes through training
    def updateLR(self, update, optimizer):
        num_updates = self.total_timesteps // self.batch_size
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * self.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    # runs steps in the environment and collects data to update the model
    def simulate_steps(self, env, agent, writer, next_obs, next_done):
        for step in range(self.num_steps):
            self.global_step += 1
            next_obs = torch.tensor(next_obs, dtype=torch.float32) if not isinstance(next_obs,
                                                                                     torch.Tensor) else next_obs.clone().detach()

            # Store current observation and done status
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # Get action, log-probability, and value
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # Execute action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward)
            next_obs = torch.Tensor(next_obs)
            next_done = torch.tensor(terminated or truncated, dtype=torch.float32)

            # Handle end of episode
            if terminated or truncated:
                print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
                next_obs, info = env.reset()
                next_done = torch.tensor(1.0, dtype=torch.float32)  # Mark as done
        return next_obs, next_done

    # computes generalized advantage estimation and returns the advantages and returns
    def compute_gae(self, agent, next_obs, next_done):
        with torch.no_grad():
            next_obs = torch.tensor(next_obs, dtype=torch.float32) if not isinstance(next_obs,
                                                                                     torch.Tensor) else next_obs.clone().detach()
            next_value = agent.get_value(next_obs).reshape(1, -1)

            # Use GAE for advantage computation
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        return advantages, returns

    # updates the policy and value networks
    def optimize_minibatch(self, agent, optimizer, mb_inds):
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(self.b_obs[mb_inds], self.b_actions.long()[mb_inds])
        logratio = newlogprob - self.b_logprobs[mb_inds]
        ratio = logratio.exp()

        mb_advantages = self.b_advantages[mb_inds]
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - self.b_returns[mb_inds]) ** 2
            v_clipped = self.b_values[mb_inds] + torch.clamp(
                newvalue - self.b_values[mb_inds],
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - self.b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - self.b_returns[mb_inds]) ** 2).mean()

        # Entropy loss and total loss
        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
        optimizer.step()

        return pg_loss, v_loss, entropy_loss
