import numpy as np
from utils import reward_player_2

def evaluate(agent, env, opponent, max_episodes=100, max_timesteps=1000, render = False, agent_is_player_1 = True):
    
    wins_per_episode = np.zeros(max_episodes + 1)
    loses_per_episode = np.zeros(max_episodes + 1)
    
    for i_episode in range(1, max_episodes + 1):
        ob, _ = env.reset()
        obs_agent2 = env.obs_agent_two()
        
        total_reward = 0
        wins_per_episode[i_episode] = 0
        loses_per_episode[i_episode] = 0
        
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
    return wins_per_episode, loses_per_episode