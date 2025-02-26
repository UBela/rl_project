import sys
import os
import torch
import time
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hockey import hockey_env as h_env
from sac_agent import SACAgent
from trainer import SACTrainer
from replay_buffer import ReplayBuffer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)


parser = argparse.ArgumentParser()

# Training options
parser.add_argument("--max_episodes", type=int, default=30000)
parser.add_argument("--max_timesteps", type=int, default=250)
parser.add_argument("--iter_fit", type=int, default=32)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--use_hard_opp", action="store_true", default=False)
parser.add_argument("--selfplay", action="store_true", default=False)
parser.add_argument("--self_play_intervall", type=int, default=50000)
parser.add_argument("--self_play_start", type=int, default=16000)
parser.add_argument("--add_self_every", type=int, default=100000)
parser.add_argument("--evaluate_every", type=int, default=100)
parser.add_argument("--results_folder", type=str, default="results")

# PER options
parser.add_argument("--use_PER", action="store_true", default=False)
parser.add_argument("--per_alpha", type=float, default=0.3)
parser.add_argument("--per_beta", type=float, default=0.4)
parser.add_argument("--per_beta_update", type=float, default=0.0006)

# SAC options
parser.add_argument("--policy_lr", type=float, default=0.001)
parser.add_argument("--q_lr", type=float, default=0.001)
parser.add_argument("--value_lr", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--automatic_entropy_tuning", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--buffer_size", type=int, default=131072)
parser.add_argument("--lr_factor", type=float, default=0.5)
parser.add_argument("--lr_milestones", nargs="+", type=int, help="Learning rate milestones")
parser.add_argument("--alpha_milestones", nargs="+", type=int, help="Alpha learning rate milestones")
parser.add_argument("--update_target_every", type=int, default=1)
parser.add_argument("--grad_steps", type=int, default=32)
parser.add_argument("--soft_tau", type=float, default=0.005)
parser.add_argument("--alpha_lr", type=float, default=1e-4, help="Learning rate for alpha parameter")

parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--show", action="store_true", default=False)
parser.add_argument("--q", action="store_true", default=False)
parser.add_argument("--evaluate", action="store_true", default=False)
parser.add_argument("--mode", type=str, default="defense", help="shooting, defense, normal")





opts = parser.parse_args()


import json
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Logger:
    """Logging class for training information."""
    
    
    def __init__(self, prefix_path="logs", mode="normal"):
        self.prefix_path = Path(prefix_path)
        self.agents_prefix_path = self.prefix_path.joinpath("agents")
        self.plots_prefix_path = self.prefix_path.joinpath("plots")
        self.arrays_prefix_path = self.prefix_path.joinpath("arrays")
        self.log_file = self.prefix_path.joinpath("training_log.json") 


        self.prefix_path.mkdir(parents=True, exist_ok=True)
        self.agents_prefix_path.mkdir(exist_ok=True)
        self.plots_prefix_path.mkdir(exist_ok=True)
        self.arrays_prefix_path.mkdir(exist_ok=True)

        
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                json.dump([], f, indent=4)

        

        

    def info(self, msg):
        """Logs single message."""
        print(msg)

    def save_model(self, agent, filename):

        model_pth = self.agents_prefix_path.joinpath(f"{filename}.pth")
        model_pkl = self.agents_prefix_path.joinpath(f"{filename}.pkl")
        
        state = {
            "policy_net": agent.policy_net.state_dict(),
            "qnet1": agent.qnet1.state_dict(),
            "qnet_target": agent.qnet_target.state_dict()
        }
        torch.save(state, model_pth)

        with open(model_pkl, "wb") as f:
            pickle.dump(agent, f)

        self.info(f"Model saved: {model_pth} & {model_pkl}")


    def log_training(self, episode, training_metrics):
        """ (SAC-Losses, Alpha, PER-Parameter) in one JSON-Datei."""
        with open(self.log_file, "r+") as f:
            logs = json.load(f)  

            log_entry = {"episode": episode}
            log_entry.update(training_metrics) 

            logs.append(log_entry) 
            f.seek(0)
            json.dump(logs, f, indent=4)  
            f.truncate()  

        self.info(f"Trainingslog saved for episode {episode}")

    def print_episode_info(self, episode, steps, reward, training_metrics=None):
        """Prints episode information."""
        log_str = f"Episode {episode} | Steps: {steps} | Reward: {reward:.3f}"

        self.info(log_str)
        self.log_training(episode, training_metrics or {})

    def print_stats(self, rewards, touch_stats, won_stats, lost_stats):
        """Printing & saving training statistics."""
        avg_reward = np.mean(rewards) if rewards else 0
        win_rate = sum(won_stats.values()) / len(won_stats) if won_stats else 0
        loss_rate = sum(lost_stats.values()) / len(lost_stats) if lost_stats else 0
        draw_rate = 1 - win_rate - loss_rate

        print(f"\n=== Training done ===")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Winrate: {win_rate:.2%} |  Lossrate: {loss_rate:.2%} | Drawrate: {draw_rate:.2%}")

        self.log_training("summary", {"avg_reward": avg_reward, "win_rate": win_rate, "loss_rate": loss_rate, "draw_rate": draw_rate})

   

    
if __name__ == "__main__":
    print("### Start preparation ###")
    print("### Options ###")
    print(opts)

    start_prep_time = time.time()
    opts.device = torch.device("cuda" if opts.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {opts.device}")

    
    mode_mapping = {
        "normal": h_env.Mode.NORMAL,
        "shooting": h_env.Mode.TRAIN_SHOOTING,
        "defense": h_env.Mode.TRAIN_DEFENSE
    }

    if opts.mode not in mode_mapping:
        raise ValueError("Unknown training mode")

    env = h_env.HockeyEnv(mode=mode_mapping[opts.mode], verbose=(not opts.q))

    
    opponents = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)]
    config = {
        "hidden_dim": [256, 256],
        "max_episodes": opts.max_episodes,
        "max_timesteps": opts.max_timesteps,
        "iter_fit": opts.iter_fit,
        "log_interval": opts.log_interval,
        "random_seed": opts.random_seed,
        "render": opts.render,
        "use_hard_opp": opts.use_hard_opp,
        "selfplay": opts.selfplay,
        "self_play_intervall": opts.self_play_intervall,
        "self_play_start": opts.self_play_start,
        "evaluate_every": opts.evaluate_every,
        "results_folder": opts.results_folder,
        "gamma": opts.gamma,
        "tau": opts.tau,
        "alpha": opts.alpha,
        "automatic_entropy_tuning": opts.automatic_entropy_tuning,
        "policy_lr": opts.policy_lr,
        "q_lr": opts.q_lr,
        "value_lr": opts.value_lr,
        "alpha_lr": opts.alpha_lr,
        "lr_milestones": opts.lr_milestones,
        "buffer_size": opts.buffer_size,
        "per_alpha": opts.per_alpha,
        "per_beta": opts.per_beta,
        "per_beta_update": opts.per_beta_update,
        "use_PER": opts.use_PER,
        "device": opts.device,
        "results_folder": opts.results_folder,
        "alpha_milestones": opts.alpha_milestones,
        "iter_fit": opts.iter_fit,
        "target_update_interval": opts.update_target_every,
        "soft_tau": opts.soft_tau,
        "batch_size": opts.batch_size,
        "cuda": opts.cuda,
        "show": opts.show,
        "q": opts.q,
        "evaluate": opts.evaluate,
        "mode": opts.mode,
        "transitions_path": opts.transitions_path,
        "add_self_every": opts.add_self_every,
        "lr_factor": opts.lr_factor,
        "update_target_every": opts.update_target_every,
        "grad_steps": opts.grad_steps
    }


    agent = SACAgent(
            state_dim=env.observation_space.shape,
            action_dim=env.action_space.shape[0] // 2,
            config=config,
            action_space=env.action_space)
  

    print("### Agent created ###")

    logger = Logger("logs")
    trainer = SACTrainer(logger, vars(opts))

    print("### Start training... ###")
    start_time = time.time()
    trainer.train(agent, opponents, env)
    print(f"Training Done! Duration: {time.time() - start_time:.2f} seconds")
