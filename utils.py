import numpy as np
import torch
import gymnasium as gym


class RandomAgent:
    
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
       
    def get_action(self, observation):
        return np.random.uniform(self._action_space.low, self._action_space.high, self._action_space.shape // 2)

    def store_transition(self):
        pass

    def state(self):
        pass