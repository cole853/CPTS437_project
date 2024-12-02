import numpy as np
import pickle


# Q learning agent
class Q_agent:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        self.q_values = {}                              # initialize Q values policy for each of the 5 actions
        self.lr = learning_rate                         # learning rate for a single simulation
        self.discount_factor = discount_factor          # discount factor for computing Q value
        self.epsilon = initial_epsilon                  # starting epsilon (probability of choosing an action randomly)
        self.epsilon_decay = epsilon_decay              # the amount to decrease epsilon after each simulation
        self.final_epsilon = final_epsilon              # the smallest the epsilon can be

    # used to get an action from the agent with probability 1-epsilon or a random action with probability epsilon
    def get_action(self, obs) -> int:
        if obs not in self.q_values.keys():
            self.q_values[obs] = [0.0] * 4
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        # with probability (1 - epsilon) use policy
        else:
            return self.q_values[obs].index(max(self.q_values[obs]))

    # update the Q-value of an action
    def update(self, obs, action, reward, terminated, next_obs):
        if obs not in self.q_values.keys():
            self.q_values[obs] = [0.0] * 4
        if next_obs not in self.q_values.keys():
            self.q_values[next_obs] = [0.0] * 4

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        return temporal_difference

    # decreases epsilon by epsilon_decay
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        return self.epsilon

    # Save the q_values dictionary as a byte stream using pickle
    def save_policy(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"  # Add '.pkl' if not already present
        with open(filename, 'wb') as f:
            pickle.dump(self.q_values, f)

    # Load the q_values dictionary from a byte stream
    def load_policy(self, filename):
        with open(filename, 'rb') as f:
            self.q_values = pickle.load(f)
