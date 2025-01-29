import numpy as np
import torch
import gymnasium as gym


class RandomAgent:
    
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self.low = self._action_space.low[0]
        self.high = self._action_space.high[0]
        self.action_shape_per_agent = self._action_space.shape[0] // 2
    def act(self, observation):
        return np.random.uniform(self.low, self.high, self.action_shape_per_agent)

    def store_transition(self):
        pass

    def state(self):
        pass