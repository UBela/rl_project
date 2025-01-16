import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gymnasium as gym
import numpy as np
from noise import OUNoise, GaussianNoise
from networks import QFunction, FeedForwardNetwork
from replay_buffer import ReplayBuffer
from td3 import TD3Agent
import pickle
import matplotlib.pyplot as plt

def main():
    rewards = []
    losses = []
    lengths = []
    timestep = 0
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    
    
    # normalize action space?
    observation_space = env.observation_space
    action_space = env.action_space
    agent = TD3Agent(observation_space, action_space)
    max_episodes = 2000
    max_time_steps = 2000
    
    
    def save_stats():
        with open(f"results_td3_{env_name}_rewards.pkl", "wb") as f:
            pickle.dump({"rewards": rewards, "losses": losses, "lengths": lengths}, f)
            
    def plot_rewards(rewards):
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.show()
        

    
    for episode in range(1, max_episodes+1):
        state, _ = env.reset()
        
        episode_reward = 0
        
        for t in range(max_time_steps):
            timestep += 1
            action = agent.get_action(state)
            (next_state, reward, done, trunc, _info) = env.step(action)
            agent.store_transition((state, action, reward, next_state, done))
            episode_reward += reward
            state = next_state
            if done or trunc:
                break
            
        losses.extend(agent.train())
        rewards.append(episode_reward)
        lengths.append(t)
        
        if episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_length = np.mean(lengths[-20:])
            
            print(f"Episode: {episode}, Avg Reward: {avg_reward}, Avg Length: {avg_length}")
    save_stats()
    
if __name__ == "__main__":
    main()