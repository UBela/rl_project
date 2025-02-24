from __future__ import annotations

import argparse
import uuid
import sys
import os
import yaml  # For loading config.yaml
import hydra


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
grand_parent_dir = os.path.dirname(parent_dir)
sys.path.append(grand_parent_dir)

from tdmpc2 import TDMPC2
import envs.hockey as h_env
import numpy as np
import torch
from comprl.client import Agent, launch_client
from common.parser import parse_cfg
from omegaconf import DictConfig, OmegaConf



agent_name = "own"


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""
    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cfg = parse_cfg(cfg)
        
        keep_mode = cfg.get("keep_mode", True)
        mode = cfg.task.split("hockey-")[-1]
        verbose = cfg.get("verbose", False)

        self.env = h_env.HockeyEnv(keep_mode=keep_mode, mode=mode, verbose=verbose)
        self.env = h_env.HockeyEnvWrapper(self.env, cfg, None)
        
        print("Initializing HockeyAgent...")  # Debugging
        self.hockey_agent = TDMPC2(cfg).to(self.device)
        self.hockey_agent.eval()
        
        checkpoint_path = "/mnt/qb/work/geiger/gwb215/test/tdmpc2/tdmpc2/checkpoints/20250223_175921_130k_only-weak-strong/final.pt"
        #checkpoint_path = "/mnt/qb/work/geiger/gwb215/test/tdmpc2/tdmpc2/logs/hockey-NORMAL/1/default/models/final.pt"
        self.hockey_agent.load(checkpoint_path)
        
    def get_step(self, observation: list[float]) -> list[float]:
        # Convert observation (list) to a PyTorch tensor and move to the correct device
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device, non_blocking=True)
        
        # Now pass it to the hockey agent's act method
        action = self.hockey_agent.act(observation_tensor).tolist()  # Convert back to list
        
        continuous_action = self.env.discrete_to_continous_action(action)

        return continuous_action

    def on_start_game(self, game_id) -> None:
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} with my score: {stats[0]} against the opponent with score: {stats[1]}")


def initialize_agent(cfg: dict, agent_args: list[str] = None) -> Agent:
    """Creates an agent using a preloaded configuration."""
    # Load Hydra config manually
    #with hydra.initialize(config_path="../../"):
    #    cfg = hydra.compose(config_name="config.yaml")
    
    agent = HockeyAgent(cfg)
    return agent

@hydra.main(config_path="../../", config_name="config.yaml", version_base=None)
def main(cfg: dict) -> None:
    """Launches the client and connects to the server."""
    launch_client(lambda agent_args: initialize_agent(cfg, agent_args))  # Pass function reference with cfg


# # Main function using Hydra to load the config from the grandparent directory
# def initialize_agent(cfg: dict, agent_args: list[str]) -> Agent:
#     #config_path = "/mnt/qb/work/geiger/gwb215/test/tdmpc2/tdmpc2/config.yaml"
#     #config_path = "/mnt/qb/work/geiger/gwb215/test/tdmpc2/tdmpc2/logs/hockey-NORMAL/1/default/wandb/run-20250216_212614-mng5nkx7/files/config.yaml"
#     #with open(config_path, "r") as file:
#     #    cfg_dict = yaml.safe_load(file)
#     
#     #print(cfg_dict)
#     #cfg = OmegaConf.create(cfg_dict)
#     
#     #cfg = parse_cfg(cfg)
# 
#     agent = HockeyAgent(cfg)
#     
#     return agent
# 
# @hydra.main(config_path="../../", config_name="config.yaml", version_base=None)
# def main(cfg: dict) -> None:
#     """Launches the client and connects to the server."""
#     launch_client(initialize_agent(cfg, agent_args=[]))








#####
# Alt:


# class RandomAgent(Agent):
#     """A hockey agent that simply uses random actions."""
#     def get_step(self, observation: list[float]) -> list[float]:
#         return np.random.uniform(-1, 1, 4).tolist()
# 
#     def on_start_game(self, game_id) -> None:
#         print("game started")
# 
#     def on_end_game(self, result: bool, stats: list[float]) -> None:
#         text_result = "won" if result else "lost"
#         print(
#             f"game ended: {text_result} with my score: "
#             f"{stats[0]} against the opponent with score: {stats[1]}"
#         )
# 
# 
# class HockeyAgent(Agent):
#     """A hockey agent that can be weak or strong."""
#     def __init__(self, cfg: dict, env: h_env.HockeyEnv) -> None:
#         super().__init__()
# 
#         checkpoint_path = "/mnt/qb/work/geiger/gwb215/test/tdmpc2/tdmpc2/checkpoints/20250219_075338_selfplay_1000000/final.pt"
#         
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.env = env
#         self.cfg = cfg
#         
#         self.hockey_agent = TDMPC2(cfg).to(self.device)
#         self.hockey_agent.load(checkpoint_path)
#         #print(self.hockey_agent)
#         self.hockey_agent.eval()
#         
#     def get_step(self, observation: list[float]) -> list[float]:
#         action = self.hockey_agent.act(observation).tolist()
#         return action
# 
#     def on_start_game(self, game_id) -> None:
#         print(f"Game started (id: {game_id})")
# 
#     def on_end_game(self, result: bool, stats: list[float]) -> None:
#         text_result = "won" if result else "lost"
#         print(f"Game ended: {text_result} with my score: {stats[0]} against the opponent with score: {stats[1]}")
# 
# @hydra.main(config_path="../../", config_name="config.yaml", version_base=None)
# # Main function using Hydra to load the config from the grandparent directory
# def initialize_agent(cfg: dict) -> Agent:
#     cfg = parse_cfg(cfg)
#     print(cfg)
#     
#     # # Create the parser
#     # parser = argparse.ArgumentParser(description="Example script to demonstrate manual argument parsing.")
# 
#     # Add arguments
#     # parser.add_argument('--server-url', type=str, help='Your server-url')
#     # parser.add_argument('--server-port', type=int, help='Your server-port')
#     # parser.add_argument('--token', type=str, help='Your token')
# 
#     # Manually fill in the arguments
#     # agent_args = parser.parse_args(['--server-url', 'comprl.cs.uni-tuebingen.de', '--server-port', '65335', '--token', '2762bbae-c4e4-47b2-86d4-aeb1b55e6a98'])
#     
#     # Initialize environment
#     keep_mode = cfg.get("keep_mode", True)
#     mode = cfg.task.split("hockey-")[-1]
#     verbose = cfg.get("verbose", False)
# 
#     env = h_env.HockeyEnv(keep_mode=keep_mode, mode=mode, verbose=verbose)
#     env = h_env.HockeyEnvWrapper(env, cfg, None)
# 
#     # Initialize the agent
#     
#     if agent_name == "own":
#         print("Initializing HockeyAgent...")  # Debugging
#         agent = HockeyAgent(cfg, env)
#     else:  # Random agent
#         print("Initializing RandomAgent...")  # Debugging
#         agent = RandomAgent()
#     if agent is None:
#         print("Agent was not properly initialized.")
#     agent = Agent(agent)
#     return agent
# 
# def main() -> None:
#     """Launches the client and connects to the server."""
#     launch_client(initialize_agent)


#########





# from __future__ import annotations
# 
# import argparse
# import uuid
# import sys
# import os
# import hydra
# 
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# grand_parent_dir = os.path.dirname(parent_dir)
# sys.path.append(grand_parent_dir)
# 
# from tdmpc2 import TDMPC2
# from common.parser import parse_cfg
# from omegaconf import DictConfig
# 
# 
# import envs.hockey as h_env
# import numpy as np
# import torch
# 
# 
# from comprl.client import Agent, launch_client
# 
# 
# class RandomAgent(Agent):
#     """A hockey agent that simply uses random actions."""
# 
#     def get_step(self, observation: list[float]) -> list[float]:
#         return np.random.uniform(-1, 1, 4).tolist()
# 
#     def on_start_game(self, game_id) -> None:
#         print("game started")
# 
#     def on_end_game(self, result: bool, stats: list[float]) -> None:
#         text_result = "won" if result else "lost"
#         print(
#             f"game ended: {text_result} with my score: "
#             f"{stats[0]} against the opponent with score: {stats[1]}"
#         )
# 
# 
# class HockeyAgent(Agent):
#     """A hockey agent that can be weak or strong."""
#     def __init__(self, cfg: DictConfig, env: h_env.HockeyEnv) -> None:
#         super().__init__()
#         
#         checkpoint_path = "/mnt/qb/work/geiger/gwb215/test/tdmpc2/tdmpc2/checkpoints/20250219_075338_selfplay_1000000/final.pt"
#         
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         #cfg = parse_cfg(cfg)  # Ensure the config is parsed correctly
#         self.env = env
#         self.cfg = cfg
#         
#         self.hockey_agent = TDMPC2(cfg).to(self.device)
#         self.model.load(checkpoint_path)
#         self.model.eval()
#         
#         # Set the environment for the agent (for interactions during evaluation)
#         self.env = env
#         
#         
#     def get_step(self, observation: list[float]) -> list[float]:
#         # NOTE: If your agent is using discrete actions (0-7), you can use
#         # HockeyEnv.discrete_to_continous_action to convert the action:
#         #
#         # from hockey.hockey_env import HockeyEnv
#         # env = HockeyEnv()
#         # continuous_action = env.discrete_to_continous_action(discrete_action)
# 
#         action = self.hockey_agent.act(observation).tolist()
#         return action
# 
#     def on_start_game(self, game_id) -> None:
#         game_id = uuid.UUID(int=int.from_bytes(game_id))
#         print(f"Game started (id: {game_id})")
# 
#     def on_end_game(self, result: bool, stats: list[float]) -> None:
#         text_result = "won" if result else "lost"
#         print(
#             f"Game ended: {text_result} with my score: "
#             f"{stats[0]} against the opponent with score: {stats[1]}"
#         )
# 
# @hydra.main(config_name='config', config_path=os.path.abspath(os.path.join(os.getcwd(), "../..")), version_base=None)
# def initialize_agent(cfg: DictConfig) -> Agent:
#     """Initialize the agent with a parsed config."""
#     
#     if not isinstance(cfg, DictConfig):
#         raise TypeError(f"Expected DictConfig, but got {type(cfg)}: {cfg}")
# 
#     # Debugging: Print the actual config
#     print(f"Loaded Config: {cfg}")
#     print(f"cfg.task: {cfg.get('task', 'NOT FOUND')}")
# 
#     if 'task' not in cfg or not cfg.task.startswith("hockey-"):
#         raise ValueError(f"Invalid task configuration: {cfg.task}")
# 
#     # Parse additional command-line arguments **separately**
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--agent",
#         type=str,
#         choices=["own", "random"],
#         default="own",
#         help="Which agent to use.",
#     )
#     args, _ = parser.parse_known_args()
# 
#     # Create Environment
#     keep_mode = cfg.get("keep_mode", True)
#     mode = cfg.task.split("hockey-")[-1]  
#     verbose = cfg.get("verbose", False)
# 
#     env = h_env.HockeyEnv(keep_mode=keep_mode, mode=mode, verbose=verbose)
#     env = h_env.HockeyEnvWrapper(env, cfg)  
# 
#     # Initialize the agent
#     agent = HockeyAgent(cfg, env) if args.agent == "own" else RandomAgent()
# 
#     return agent
# 
# 
# def main() -> None:
#     """Launches the client with Hydra's config initialization."""
#     launch_client(lambda _: initialize_agent())  # Ensure `initialize_agent` is called correctly
# 

# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
# @hydra.main(config_name='config', config_path='../..', version_base=None)
# def initialize_agent(cfg: DictConfig) -> Agent:
#     # Use argparse to parse the arguments given in `agent_args`.
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--agent",
#         type=str,
#         choices=["own", "random"],
#         default="own",
#         help="Which agent to use.",
#     )
#     args, unknown = parser.parse_known_args()
# 
#     print(f"Loaded Config: {cfg}")
#     print(f"cfg.task: {cfg.get('task', 'NOT FOUND')}")
# 
#     cfg = parse_cfg(cfg)
#     # Initialize environment
#     #env = h_env.make_env(cfg, agent)
#     
#     #env = h_env.HockeyEnv(cfg)  # Create base environment
#     #env = HockeyEnvWrapper(env, cfg)
#     
#     keep_mode = cfg.get("keep_mode", True)
#     mode = cfg.task.split("hockey-")[-1]  
#     verbose = cfg.get("verbose", False)
#     
#     env = h_env.HockeyEnv(keep_mode=keep_mode, mode=mode, verbose=verbose)
#     env = h_env.HockeyEnvWrapper(env, cfg, agent)
# 
# 
#     # Initialize the agent based on the arguments.
#     agent: Agent
#     if args.agent == "own":
#         agent = HockeyAgent(cfg, env)
#     #elif args.agent == "strong":
#         #agent = HockeyAgent(weak=False)
#     elif args.agent == "random":
#         agent = RandomAgent()
#     else:
#         raise ValueError(f"Unknown agent: {args.agent}")
# 
#     # And finally return the agent.
#     return agent
# 
# 
# def main() -> None:
#     launch_client(initialize_agent)


if __name__ == "__main__":
    main()
