import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from td3.evaluate import evaluate
from td3.td3_agent import TD3Agent
from hockey import hockey_env as h_env


def main(state_dict_path, max_episodes):
    print("### Start preparation ###")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")   
    
    env = h_env.HockeyEnv(h_env.Mode.NORMAL)
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False) 
    opponents = [weak_opponent, strong_opponent]
    
    agent = TD3Agent(env.observation_space, env.action_space, device, {})
    agent.restore_state(torch.load(state_dict_path))
    agent.set_to_eval()
    assert agent.in_training == False
    
    print("### Preparation done ###")
    print("### Start evaluation ###")
    
    for opponent in opponents:
        wins, loses = evaluate(agent, env, opponent, max_episodes=max_episodes, max_timesteps=1000, render=True, agent_is_player_1=True)
        win_rate, lose_rate = sum(wins)/max_episodes, sum(loses)/max_episodes
        print(f"Winrate: {win_rate:.3f} Lossrate: {lose_rate:.3f}")
    
    print("### Evaluation done ###")
    print("### End ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict_path', type=str, required=True, help='Path to the trained model state dict')
    parser.add_argument('--max_episodes', type=int, default=4, help='Number of games to play against each opponent')
    args = parser.parse_args()
    
    main(args.state_dict_path, args.max_episodes)
