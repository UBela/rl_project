import sys
import os
import torch
import time
import random
import argparse
from hockey import hockey_env as h_env
from sac_agent import SACAgent
from trainer import SACTrainer
import numpy as np
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer

# Sicherstellen, dass das Skript das richtige Verzeichnis nutzt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Erstelle benötigte Verzeichnisse
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Argumente für Flexibilität beim Training
parser = argparse.ArgumentParser()

# Trainingsparameter
# OptParser für Trainingsoptionen


# Trainingsparameter
parser.add_argument("--max_episodes", type=int, default=25000)
parser.add_argument("--max_timesteps", type=int, default=1000)
parser.add_argument("--iter_fit", type=int, default=32)
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--use_hard_opp", action="store_true", default=False)
parser.add_argument("--selfplay", action="store_true", default=False)  # EINDEUTIGE Benennung
parser.add_argument("--self_play_intervall", type=int, default=50000)
parser.add_argument("--self_play_start", type=int, default=16000)
parser.add_argument("--evaluate_every", type=int, default=2000)
parser.add_argument("--results_folder", type=str, default="results")

# Prioritized Experience Replay (PER)
parser.add_argument("--use_PER", action="store_true", default=False)
parser.add_argument("--per_alpha", type=float, default=0.6)
parser.add_argument("--per_beta", type=float, default=0.4)
parser.add_argument("--per_beta_update", type=float, default=None)

# SAC Hyperparameter
parser.add_argument("--policy_lr", type=float, default=1e-4)
parser.add_argument("--q_lr", type=float, default=1e-3)
parser.add_argument("--value_lr", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--automatic_entropy_tuning", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--buffer_size", type=int, default=131072)

# Zusätzliche Optionen aus funktionierender `train.py`
parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--show", action="store_true", default=False)
parser.add_argument("--q", action="store_true", default=False)
parser.add_argument("--evaluate", action="store_true", default=False)
parser.add_argument("--mode", type=str, default="defense", help="(shooting | defense | normal)")
parser.add_argument("--preload_path", type=str, default=None)
parser.add_argument("--transitions_path", type=str, default=None)
parser.add_argument("--add_self_every", type=int, default=100000)

# Lernraten-Anpassung
parser.add_argument("--lr_factor", type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+', type=int)
parser.add_argument('--alpha_milestones', help='Alpha learning rate milestones', nargs='+', type=int)
parser.add_argument("--update_target_every", type=int, default=1)
parser.add_argument("--grad_steps", type=int, default=32)
parser.add_argument("--soft_tau", type=float, default=0.005)
parser.add_argument("--alpha_lr", type=float, default=1e-4, help="Lernrate für die Alpha-Anpassung")



opts = parser.parse_args()

import os
import json

class SimpleLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.episode_log_file = os.path.join(log_dir, "episode_log.json")  
        self.training_log_file = os.path.join(log_dir, "training_log.json")  
        self.plot_dir = log_dir

        # Falls die JSONs nicht existieren, leere Listen schreiben
        for file in [self.episode_log_file, self.training_log_file]:
            if not os.path.exists(file):
                with open(file, "w") as f:
                    json.dump([], f, indent=4)

    def log(self, message):
        """Loggt Nachrichten zur Konsole"""
        print(message)

    def log_episode(self, episode, steps, reward, winner):
        """Speichert Episoden-Infos (Reward, Winner)"""
        with open(self.episode_log_file, "r+") as f:
            logs = json.load(f)
            logs.append({
                "episode": episode,
                "steps": steps,
                "reward": reward,
                "winner": winner
            })
            f.seek(0)
            json.dump(logs, f, indent=4)
            f.truncate()

    def log_training(self, episode, training_metrics):
        """Speichert Trainingsparameter (SAC-Losses, Alpha)"""
        with open(self.training_log_file, "r+") as f:
            logs = json.load(f)
            log_entry = {"episode": episode}
            log_entry.update(training_metrics)

            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=4)
            f.truncate()

    def save_model(self, agent, filename_base):
        """Speichert das Modell als .pkl und .pth"""
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)

        pkl_path = os.path.join(model_dir, f"{filename_base}.pkl")
        torch.save(agent, pkl_path)

        pth_path = os.path.join(model_dir, f"{filename_base}.pth")
        torch.save(agent.policy_net.state_dict(), pth_path)

        self.log(f"✅ Modell gespeichert: {pkl_path} & {pth_path}")

    def print_episode_info(self, episode, steps, reward, winner, training_metrics=None):
        """Druckt & speichert Episoden-Infos"""
        log_str = (
            f"Episode {episode} | Steps: {steps} | Reward: {reward:.3f} | Winner: {winner}"
        )
        if training_metrics:
            log_str += f" | Alpha: {training_metrics.get('alpha', 'N/A'):.4f} | Q1_Loss: {training_metrics.get('q1_loss', 'N/A'):.4f}"
        
        self.log(log_str)
        self.log_episode(episode, steps, reward, winner)
        
        if training_metrics:
            self.log_training(episode, training_metrics)

    def plot_running_mean(self, data, title, filename, window_size=100):
        """Plottet den gleitenden Durchschnitt der Rewards"""
        if len(data) == 0:
            print("[WARNUNG] Keine Daten zum Plotten.")
            return

        running_mean = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        plt.figure(figsize=(10, 5))
        plt.plot(running_mean, label="Running Mean Reward", color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Reward-Verlauf gespeichert: {plot_path}")


if __name__ == "__main__":
    print("### Start preparation ###")
    print("### Options ###")
    print(opts)

    start_prep_time = time.time()
    opts.device = torch.device("cuda" if opts.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {opts.device}")

    # Wähle Modus für Training
    if opts.mode == "normal":
        mode = h_env.Mode.NORMAL
    elif opts.mode == "shooting":
        mode = h_env.Mode.TRAIN_SHOOTING
    elif opts.mode == "defense":
        mode = h_env.Mode.TRAIN_DEFENSE
    else:
        raise ValueError("Unknown training mode. Use: --mode (shooting | defense | normal)")

    
    env = h_env.HockeyEnv(mode=mode, verbose=(not opts.q))
    
    # Erstelle Gegner
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False)
    opponents = [weak_opponent, strong_opponent]

    # Erstelle Replay Buffer
    replay_buffer = ReplayBuffer(opts.buffer_size)

    # Initialisiere SAC-Agent basierend auf den richtigen Parametern
    if opts.preload_path is None:
        agent = SACAgent(
            state_dim=env.observation_space.shape[0],  # 🟢 `state_dim` statt `obs_dim`
            action_dim=env.action_space.shape[0] // 2,
            hidden_dim=256,  # Standardwert, falls nicht per CLI gesetzt
            gamma=opts.gamma,
            tau=opts.tau,
            alpha=opts.alpha,
            automatic_entropy_tuning=opts.automatic_entropy_tuning,
            policy_lr=opts.policy_lr,
            q_lr=opts.q_lr,
            value_lr=opts.value_lr,
            alpha_lr=opts.alpha_lr,
            buffer_size=opts.buffer_size,
            per_alpha=opts.per_alpha,
            per_beta=opts.per_beta,
            per_beta_update=opts.per_beta_update,
            use_PER=opts.use_PER,
            device=opts.device,
            results_folder=opts.results_folder,
            alpha_milestones=opts.alpha_milestones
        )
    else:
        print(f"⚠ Lade gespeichertes Modell aus {opts.preload_path}")
        agent = SACAgent.load_model(opts.preload_path)

    print("### Agent created ###")

    logger = SimpleLogger("logs")
    config = vars(opts)
    config["use_self_play"] = opts.selfplay
    trainer = trainer = SACTrainer(logger, config)

    
    print("### Trainer created ###")
    print("### Start training... ###")
    end_prep_time = time.time()
    print(f"Preparation time: {end_prep_time - start_prep_time:.2f} seconds.")

    # Training starten
    start_time = time.time()
    trainer.train(agent,opponents,env)
    end_time = time.time()

    print(f"✅ Training abgeschlossen! Dauer: {end_time - start_time:.2f} Sekunden")
