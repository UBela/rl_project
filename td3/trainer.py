import torch
from td3 import TD3
import os 
import gymnasium as gym
import numpy as np
from hockey import HockeyEnv as h_env
import optparse
import pickle


class TD3Trainer:
    
    def __init__(self, config):
        self.config = config
    
    def _save_statistics(self, rewards, lengths, losses, wins_per_episode, loses_per_episode, train_iter):
        with open(f"results/results_td3_t{train_iter}_stats.pkl", "wb") as f:
            pickle.dump({"rewards": rewards, "lengths": lengths, "losses": losses, "wins": wins_per_episode, "loses": loses_per_episode}, f)
    
    def _select_opponent(self, opponents: list, episode: int):
        """Selects an opponent based on the current episode number.

        The opponent selection follows these rules:
        - A random opponent is selected for the first `episodes_on_random` episodes.
        - A weak opponent is selected for the next `episodes_on_weak` episodes.
        - A strong opponent is selected for the remaining episodes.

        Args:
            opponents (list): A list of opponent agents [random, weak, strong].
            episode (int): The current episode number.

        Returns:
            object: The selected opponent agent.
        """
        if episode <= self.config['episodes_on_random']:
            return opponents[0]
        elif episode <= self.config['episodes_on_weak']:
            return opponents[1]
        else:
            return opponents[2]

    def train(self, agent, opponents, env):
        
        iter_fit = self.config['iter_fit']
        log_interval = self.config['log_interval']
        max_episodes = self.config['max_episodes']
        max_timesteps = self.config['max_timesteps']

        random_seed = self.config['random_seed']
        
        rewards = []
        lengths = []
        wins_per_episode = {}
        loses_per_episode = {}
        losses = []
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        for i_episode in range(1, max_episodes + 1):
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            total_reward = 0
            wins_per_episode[i_episode] = 0
            loses_per_episode[i_episode] = 0
            
            opponent = self._select_opponent(opponents, i_episode)
                
            for t in range(max_timesteps):
                
                a1 = agent.get_action(ob)
                a2 = opponent.get_action(obs_agent2)
                (ob_new, reward, done, trunc, _info) = env.step(np.hstack[a1, a2])  
                
                agent.store_transition((ob, a1, reward, ob_new, done))
                total_reward += reward
                ob = ob_new
                obs_agent2 = env.obs_agent_two()
                if done or trunc:
                    wins_per_episode[i_episode] = 1 if _info['winner'] == 1 else 0
                    loses_per_episode[i_episode] = 1 if _info['winner'] == -1 else 0
                    break
            
            losses.extend(agent.train(iter_fit=iter_fit))
            rewards.append(total_reward)
            lengths.append(t)
            
            if i_episode % 500 == 0:
                print("########## Saving a checkpoint... ##########")
                torch.save(agent.state(), f'./results/td3_{i_episode}-t{iter_fit}-s{random_seed}.pth')
                self._save_statistics(rewards, lengths, losses, wins_per_episode, loses_per_episode, iter_fit)
                
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
                # print wins and loses
                print(f"Total Wins: {sum(wins_per_episode.values())} Total Loses: {sum(loses_per_episode.values())}")
                # average wins and loses
                print(f"Average Wins: {np.mean(list(wins_per_episode.values())):.3f} Average Loses: {np.mean(list(loses_per_episode.values())):.3f}")
                
                
            
            
            