import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import gymnasium as gym
import imageio

class RandomAgent:
    
    def __init__(self, action_space):
        self._action_space = action_space
        self.low = self._action_space.low[0]
        self.high = self._action_space.high[0]
        self.action_shape_per_agent = self._action_space.shape[0] // 2
    def act(self):
        return np.random.uniform(self.low, self.high, self.action_shape_per_agent)

    def store_transition(self):
        pass

    def state(self):
        pass
    

# source: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, path='./', filename='hockey_game.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=10)
    anim.save(path + filename, writer='imagemagick', fps=60)
