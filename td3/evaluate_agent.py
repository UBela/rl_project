import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from td3.evaluate import evaluate
from td3.td3_agent import TD3Agent
import torch
from hockey import hockey_env as h_env

if __name__ == '__main__':
    print("### Start preparation ###")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")   
    env = h_env.HockeyEnv(h_env.Mode.NORMAL)
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False) 
    opponents = [weak_opponent, strong_opponent]
    agent = TD3Agent(env.observation_space, env.action_space, device, {})
    state_dict_path="./results/competition/update_every_5_tau_0.005_lr_50_sp_30/td3_60000-t32-sNone.pth"
    agent.restore_state(torch.load(state_dict_path))
    agent.set_to_eval()
    assert agent.in_training == False
    print("### Preparation done ###")
    print("### Start evaluation ###")
    #play one game per opponent and record the results
    for opponent in opponents:
        max_episodes = 4
        wins, loses = evaluate(agent, env, opponent, max_episodes=max_episodes, max_timesteps=1000, render=True, agent_is_player_1=True)
        win_rate, lose_rate = sum(wins)/max_episodes, sum(loses)/max_episodes
        print(f"Winrate: {win_rate:.3f} Lossrate: {lose_rate:.3f}")
    print("### Evaluation done ###")
    print("### End ###")
