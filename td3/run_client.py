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

      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = h_env.HockeyEnv(h_env.Mode.NORMAL)
        self.agent = TD3Agent(observation_space=self.env.observation_space, 
                            action_space=self.env.action_space, 
                            device=self.device,
                            userconfig={})
        self.agent.restore_state(state)
        self.agent.set_to_eval()
        assert self.agent.in_training == False
        
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
    state_dict_path="td3/results/competition/update_every_5_tau_0.005_lr_50_sp_30/td3_60000-t32-sNone.pth"
    # Initialize the agent based on the arguments.
    agent = HockeyAgent(state=torch.load(state_dict_path))

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()