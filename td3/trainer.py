import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import torch
import time
import numpy as np
import pickle
from td3.utils import *
from td3.evaluate import evaluate
from hockey import hockey_env as h_env
import copy
#from pink import PinkActionNoise

class TD3Trainer:
    
    def __init__(self, config):
        self.config = config
        self.total_gradient_steps = 0
        self.total_steps = 0
        
    def _save_statistics(self, rewards, lengths, losses, wins_per_episode, loses_per_episode, train_iter, eval_wins_easy, eval_loses_easy, eval_wins_hard, eval_loses_hard):
        if not self.config['use_hard_opp']:
            eval_wins_hard = None
            eval_loses_hard = None
            
        with open(f"{self.config['results_folder']}/results_td3_t{train_iter}_stats.pkl", "wb") as f:
            pickle.dump({"rewards": rewards, "lengths": lengths, "losses": losses, 
                         "wins": wins_per_episode, "loses": loses_per_episode, 
                         "eval_wins_easy": eval_wins_easy, "eval_loses_easy": eval_loses_easy,
                         "eval_wins_hard": eval_wins_hard, "eval_loses_hard": eval_loses_hard}, f)
    
    def _select_opponent(self, opponents: list):
        return np.random.choice(opponents)
        

    def _add_self_play_agent(self, agent, opponents, i_episode):
        if not self.config['use_self_play']:
            return
        if i_episode >= self.config['self_play_start'] and self.total_gradient_steps % self.config['self_play_intervall'] == 0:
            print("Adding agent to self-play opponents...")
            # add deep copy of agent to opponents
            opponents.append(copy.deepcopy(agent))
    
    def fill_replay_buffer(self, agent, env):  
        
        while len(agent.replay_buffer) < self.config['buffer_size']:
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            opponent = self._select_opponent(opponents=[h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)])
            done = False
            trunc = False
            while not (done or trunc):
                a1 = np.random.uniform(-1, 1, 4)
                a2 = opponent.act(obs_agent2)
                actions = np.hstack([a1, a2])
                (ob_new, reward, done, trunc, _info) = env.step(actions)
                
                agent.store_transition((ob, a1, reward, ob_new, done))
                ob = ob_new
                obs_agent2 = env.obs_agent_two()
         

    def train(self, agent, opponents, env):
        
    
        iter_fit = self.config['iter_fit']
        log_interval = self.config['log_interval']
        max_episodes = self.config['max_episodes']
        max_timesteps = self.config['max_timesteps']
        render = self.config['render']
        random_seed = self.config['random_seed']
        use_hard_opp = self.config['use_hard_opp']
        evaluate_every = self.config['evaluate_every']         
      
        rewards = []
        lengths = []
        wins_per_episode = {}
        loses_per_episode = {}
        losses = []
        wins_per_episode_eval_easy = {}
        loses_per_episode_eval_easy = {}
        wins_per_episode_eval_hard = {}
        loses_per_episode_eval_hard = {}
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            env.seed(random_seed)
        
        #pink_noise = PinkActionNoise(sigma=0.3, seq_len=max_timesteps, action_dim=4)
        #print("Pink noise: ", pink_noise.shape)
        
        if self.config['use_PER']:
            print("Filling replay buffer...")
            self.fill_replay_buffer(agent, env)
            print("Replay buffer filled.")
           
        for i_episode in range(1, max_episodes + 1):
            # 50% chance that the agent is player 1
            agent_is_player_1 = np.random.choice([True, False])
            agent_is_player_1 = True
            
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            
            total_reward = 0
            wins_per_episode[i_episode] = 0
            loses_per_episode[i_episode] = 0
            
            if self.config['use_self_play']:
                opponent = self._select_opponent(opponents)
            else:
                opponent = opponents[1]
            
            if self.config['use_PER']:
                if self.config['per_beta_update'] is None:
                    beta_update = (1.0 - self.config["per_beta"]) / max_episodes
                else: 
                    beta_update = self.config['per_beta_update']
                agent.update_per_beta(beta_update)
            
            for t in range(max_timesteps):
                
                if agent_is_player_1:
                    a1 = agent.act(ob)
                    a2 = opponent.act(obs_agent2)
                    actions = np.hstack([a1, a2])
                else:
                    a1 = opponent.act(obs_agent2)
                    a2 = agent.act(ob)
                    actions = np.hstack([a1, a2])
                
                (ob_new, reward, done, trunc, _info) = env.step(actions)
                #if render: env.render()
                
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

                self.total_steps += 1

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
            
            self.total_gradient_steps += iter_fit
                
            self._add_self_play_agent(agent, opponents, i_episode)
            if i_episode % 500 == 0:
                print("########## Saving a checkpoint... ##########")
                torch.save(agent.state(), f'{self.config['results_folder']}/td3_{i_episode}-t{iter_fit}-s{random_seed}.pth')
                
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
                # Winrate and Lossrate
                print(f"Winrate: {sum(wins_per_episode.values())/i_episode:.3f} Lossrate: {sum(loses_per_episode.values())/i_episode:.3f}")
                
            if i_episode % evaluate_every == 0 or i_episode == max_episodes:
                agent.set_to_eval()
                print("########## Evaluating agent...########## ")
                print("Against weak opponent...")
                wins, loses = evaluate(agent, env, opponents[0], max_episodes=100, max_timesteps=1000, render=False, agent_is_player_1=agent_is_player_1)
                win_rate, lose_rate = sum(wins)/100, sum(loses)/100
                print(f"Winrate: {win_rate:.3f} Lossrate: {lose_rate:.3f}")
                wins_per_episode_eval_easy[i_episode] = win_rate
                loses_per_episode_eval_easy[i_episode] = lose_rate
                if use_hard_opp:
                    print("Against strong opponent...")
                    wins, loses = evaluate(agent, env, opponents[1], max_episodes=100, max_timesteps=1000, render=False, agent_is_player_1=agent_is_player_1)
                    win_rate, lose_rate = sum(wins)/100, sum(loses)/100
                    print(f"Winrate: {win_rate:.3f} Lossrate: {lose_rate:.3f}")
                    wins_per_episode_eval_hard[i_episode] = win_rate
                    loses_per_episode_eval_hard[i_episode] = lose_rate
                self._save_statistics(rewards, lengths, losses, wins_per_episode, loses_per_episode, iter_fit, 
                                      wins_per_episode_eval_easy, loses_per_episode_eval_easy, 
                                      wins_per_episode_eval_hard, loses_per_episode_eval_hard)

            agent.set_to_train()
