from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import uuid
import optparse

import hockey.hockey_env as h_env
import numpy as np
from sac_agent import SACAgent
from comprl.client import Agent, launch_client
import torch


class HockeyAgent(Agent):
    def __init__(self, state) -> None:
        super().__init__()

      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = h_env.HockeyEnv(h_env.Mode.NORMAL)

        config = {
                "max_episodes": 40000,
                "max_timesteps": 250,
                "iter_fit": 32,
                "log_interval": 10,
                "random_seed": None,
                "render": False,
                "use_hard_opp": False,
                "selfplay": True,
                "self_play_intervall": 50000,
                "self_play_start": 16000,
                "evaluate_every": 1000,
                "results_folder": "results",
                "hidden_dim": [256, 256],
                
                # Prioritized Experience Replay (PER)
                "use_PER": True,
                "per_alpha": 0.6,
                "per_beta": 0.4,
                "per_beta_update": None,

                # SAC Hyperparameter
                "policy_lr": 0.001,
                "q_lr": 0.001,
                "value_lr": 0.001,
                "tau": 0.005,
                "gamma": 0.95,
                "alpha": 0.2,
                "automatic_entropy_tuning": True,
                "alpha_lr": 0.0001,

                # Training Settings
                "batch_size": 128,
                "buffer_size": 131072,
                "cuda": False,
                "show": False,
                "q": False,
                "evaluate": False,
                "mode": "normal",
                "preload_path": None,
                "transitions_path": None,
                "add_self_every": 100000,

                # Lernraten-Anpassung
                "lr_factor": 0.5,
                "lr_milestones": [10000, 18000],
                "alpha_milestones": [10000, 18000],
                "update_target_every": 1,
                "grad_steps": 32,
                "soft_tau": 0.005
            }
        self.agent = SACAgent(state_dim=self.env.observation_space.shape[0], action_space=self.env.action_space, config=config)
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
    state_dict_path=r"C:\Users\regin\Documents\Winter24_25\rl_project\sac\saved_models\19570.pth"
    # Initialize the agent based on the arguments.
    agent = HockeyAgent(state=torch.load(state_dict_path, map_location='cpu'))

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()