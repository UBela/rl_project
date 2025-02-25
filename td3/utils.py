import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

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
    
    

def reward_player_2(env):
        """Reverse the hockey_env reward formula:
        r = 0
        if self.done:
            if self.winner == 0:  # tie
                r += 0
            elif self.winner == 1:  # you won
                r += 10
            else:  # opponent won
                r -= 10
        return float(r)

        Args:
            env (_type_): _description_
        """
        r = 0.0
        if env.done:
            if env.winner == 0:
                r += 0.0
            elif env.winner == 1:
                r -= 10.0
            else:
                r += 10.0
        return r
    

def get_frame(env):
    fig, ax = plt.subplots(figsize=(6,6))
    env.render(mode='human')
    plt.axis('off') 
    
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame
def frames_2_gif(frames, filename):
    if len(frames) == 0:
        return
    imageio.mimsave(filename, frames, fps=30)
