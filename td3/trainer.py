import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import torch
import numpy as np
import pickle
from td3.utils import *
from evaluate import evaluate
#from pink import PinkActionNoise

class TD3Trainer:
    
    def __init__(self, config):
        self.config = config
    
    def _save_statistics(self, rewards, lengths, losses, wins_per_episode, loses_per_episode, train_iter):
        with open(f"td3/results/results_td3_t{train_iter}_stats.pkl", "wb") as f:
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
        elif episode <= self.config['episodes_on_weak'] + self.config['episodes_on_random']:
            return opponents[1]
        else:
            return opponents[2]
        
    
    

    def train(self, agent, opponents, env):
        
        iter_fit = self.config['iter_fit']
        log_interval = self.config['log_interval']
        max_episodes = self.config['max_episodes']
        max_timesteps = self.config['max_timesteps']
        render = self.config['render']
        random_seed = self.config['random_seed']
        curriculum_learning = self.config['use_curr_learning']  
        use_hard_opp = self.config['use_hard_opp']
        evaluate_every = self.config['evaluate_every']
        
        self_play_buffer = []
        rewards = []
        lengths = []
        wins_per_episode = {}
        loses_per_episode = {}
        losses = []
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            env.seed(random_seed)
        
        #pink_noise = PinkActionNoise(sigma=0.3, seq_len=max_timesteps, action_dim=4)
        #print("Pink noise: ", pink_noise.shape)

        for i_episode in range(1, max_episodes + 1):
            # 50% chance that the agent is player 1
            agent_is_player_1 = np.random.choice([True, False])
            
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            
            total_reward = 0
            wins_per_episode[i_episode] = 0
            loses_per_episode[i_episode] = 0
            if curriculum_learning:
                opponent = self._select_opponent(opponents, i_episode)
            elif use_hard_opp:
                opponent = opponents[2]
            else:
                opponent = opponents[1]
            
            
            for t in range(max_timesteps):
                
                if agent_is_player_1:
                    a1 = agent.get_action(ob)
                    a2 = opponent.act(obs_agent2)
                    actions = np.hstack([a1, a2])
                else:
                    a1 = opponent.act(obs_agent2)
                    a2 = agent.get_action(ob)
                    actions = np.hstack([a1, a2])
                
                (ob_new, reward, done, trunc, _info) = env.step(actions)
                if render: env.render()
                
                reward = reward_player_2(env) if not agent_is_player_1 else reward

                if agent_is_player_1:
                    agent.store_transition((ob, a1, reward, ob_new, done))
                else:
                    agent.store_transition((ob, a2, reward, ob_new, done))
                
                total_reward += reward
                ob_new_copy = ob_new  
                if agent_is_player_1:
                    ob = ob_new
                    obs_agent2 = env.obs_agent_two()
                else:
                    ob = ob_new_copy  
                    obs_agent2 = env.obs_agent_two()


                if done or trunc:
                    winner = _info.get('winner', None)
                    if agent_is_player_1:
                        wins_per_episode[i_episode] = 1 if winner == 1 else 0
                        loses_per_episode[i_episode] = 1 if winner == -1 else 0
                    else:
                        wins_per_episode[i_episode] = 1 if winner == -1 else 0
                        loses_per_episode[i_episode] = 1 if winner == 1 else 0
                    break
            
            losses.extend(agent.train(iter_fit=iter_fit))
            rewards.append(total_reward)
            lengths.append(t)
            
            if i_episode % 500 == 0:
                print("########## Saving a checkpoint... ##########")
                torch.save(agent.state(), f'td3/results/td3_{i_episode}-t{iter_fit}-s{random_seed}.pth')
                self._save_statistics(rewards, lengths, losses, wins_per_episode, loses_per_episode, iter_fit)
                
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
                # Winrate and Lossrate
                print(f"Winrate: {sum(wins_per_episode.values())/i_episode:.3f} Lossrate: {sum(loses_per_episode.values())/i_episode:.3f}")
                
            if i_episode % evaluate_every == 0:
                agent.set_to_eval()
                print("########## Evaluating agent...########## ")
                print("Against weak opponent...")
                wins, loses = evaluate(agent, env, opponents[1], max_episodes=100, max_timesteps=1000, render=False, agent_is_player_1=agent_is_player_1)
                print(f"Winrate: {sum(wins)/100} Lossrate: {sum(loses)/100}")
                if use_hard_opp:
                    print("Against strong opponent...")
                    wins, loses = evaluate(agent, env, opponents[2], max_episodes=100, max_timesteps=1000, render=False, agent_is_player_1=agent_is_player_1)
                    print(f"Winrate: {sum(wins)/100:.3f} Lossrate: {sum(loses)/100:.3f}")
            
            agent.set_to_train()