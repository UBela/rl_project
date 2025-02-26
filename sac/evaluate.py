import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')
import os
import numpy as np
from td3.utils import reward_player_2  
from contextlib import redirect_stdout

def evaluate(agent, env, opponent, max_episodes=100, max_timesteps=1000, render=False, agent_is_player_1=True):
    
    wins_per_episode = np.zeros(max_episodes + 1)
    loses_per_episode = np.zeros(max_episodes + 1)
    touch_stats = np.zeros(max_episodes + 1) 
    reward_stats = np.zeros(max_episodes + 1)  
    
    for i_episode in range(1, max_episodes + 1):
        ob, _ = env.reset()
        obs_agent2 = env.obs_agent_two()
        
        with open(os.devnull, 'w') as f, redirect_stdout(f): 
            total_reward = 0
            wins_per_episode[i_episode] = 0
            loses_per_episode[i_episode] = 0
            touch_stats[i_episode] = 0
            
            for t in range(max_timesteps):
                if agent_is_player_1:
                    a1 = agent.select_action(ob)  
                    a2 = opponent.act(obs_agent2)
                    actions = np.hstack([a1, a2])
                else:
                    a1 = opponent.act(obs_agent2)
                    a2 = agent.select_action(ob)  
                    actions = np.hstack([a1, a2])

                (ob_new, reward, done, trunc, _info) = env.step(actions)
                if render: env.render()
                
                reward = reward_player_2(env) if not agent_is_player_1 else reward

                total_reward += reward
                touch_stats[i_episode] = max(touch_stats[i_episode], _info.get('reward_touch_puck', 0.0)) 

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
            
            reward_stats[i_episode] = total_reward 

    return reward_stats, touch_stats, wins_per_episode, loses_per_episode

