import os
import sys
import argparse
import torch
import hockey.hockey_env as h_env
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from trainer import Trainer

# Argumente für das Training
parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=5000, help="Maximale Episoden für das Training")
parser.add_argument('--max_steps', type=int, default=250, help="Maximale Schritte pro Episode")
parser.add_argument('--batch_size', type=int, default=128, help="Batch Size für das Training")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="Lernrate")
parser.add_argument('--gamma', type=float, default=0.99, help="Discount Faktor")
parser.add_argument('--tau', type=float, default=0.005, help="Soft Update Faktor")
parser.add_argument('--evaluate_every', type=int, default=1000, help="Evaluation nach X Episoden")
parser.add_argument('--save_model', type=str, default="saved_models/sac_model.pth", help="Pfad zum Speichern des Modells")
args = parser.parse_args()

# Umgebung initialisieren
env = h_env.HockeyEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Replay Buffer
replay_buffer = ReplayBuffer(max_size=1000000)

# SAC-Agent
agent = SACAgent(state_dim, action_dim, gamma=args.gamma, tau=args.tau, lr=args.learning_rate)

# Trainer
trainer = Trainer(agent, env, replay_buffer, args)
trainer.train()

# Modell speichern
torch.save(agent.policy_net.state_dict(), args.save_model)
print(f"✅ Modell gespeichert unter: {args.save_model}")
