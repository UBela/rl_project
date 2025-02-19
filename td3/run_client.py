from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import uuid
import optparse

import hockey.hockey_env as h_env
import numpy as np
from td3.td3_agent import TD3Agent
from comprl.client import Agent, launch_client
import torch


class HockeyAgent(Agent):
    def __init__(self, state) -> None:
        super().__init__()
        optParser = optparse.OptionParser()

        # Training parameters
        optParser.add_option("--max_episodes", type=int, default=30_000)
        optParser.add_option("--max_timesteps", type=int, default=1000)
        optParser.add_option("--iter_fit", type=int, default=32)
        optParser.add_option("--log_interval", type=int, default=20)
        optParser.add_option("--random_seed", type=int, default=None)
        optParser.add_option("--render", action="store_true", default=False)
        optParser.add_option("--use_hard_opp", action="store_true", default=True)
        optParser.add_option("--use_self_play", action="store_true", default=True)
        optParser.add_option("--self_play_intervall", type=int, default=50_000)
        optParser.add_option("--self_play_start", type=int, default=16_000)
        optParser.add_option("--evaluate_every", type=int, default=2000)
        optParser.add_option("--results_folder", type=str, default="/home/stud311/work/rl_project/td3/results")
        optParser.add_option("--use_PER", action="store_true", default=True)
        optParser.add_option("--per_alpha", help="Alpha for PER", type=float, default=0.3)
        optParser.add_option("--per_beta", help="Beta for PER", type=float, default=0.4)
        optParser.add_option("--per_beta_update", help="Beta update for PER", type=float, default=0.0006)
        # agent parameters
        optParser.add_option("--actor_lr", type=float, default=1e-4)
        optParser.add_option("--critic_lr", type=float, default=1e-3)
        optParser.add_option("--tau", type=float, default=0.005)
        optParser.add_option("--gamma", type=float, default=0.99)
        optParser.add_option("--noise_std", type=float, default=0.2)
        optParser.add_option("--noise_clip", type=float, default=0.5)
        optParser.add_option("--batch_size", type=int, default=128)
        optParser.add_option("--buffer_size", type=int, default=int(2**20))
        optParser.add_option("--policy_update_freq", type=int, default=2)
        opts, _ = optParser.parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = h_env.HockeyEnv(h_env.Mode.NORMAL)
        self.agent = TD3Agent(obersevation_space=self.env.observation_space, 
                            action_space=self.env.action_space, 
                            device=self.device,
                            userconfig=vars(opts))
        self.agent.restore_state(state)
        self.agent.set_to_eval_mode()
        
    def get_step(self, observation: list[float]) -> list[float]:

        action = self.agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )
    
    
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state_path",
        type=str,
        default='./results/both/per_best_update_every_5/td3_30000-t32-sNone.pth',
        help="Path to state dict of the agent",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent = HockeyAgent(args.state_path)

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()