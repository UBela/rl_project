import os
import sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../td3')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import hockey.hockey_env as h_env
from utils.replay_buffer import ReplayBuffer
from td3.evaluate import evaluate_agent
from sac_agent import SACAgent
from td3.td3_agent import TD3Agent

# Argumente für das Training
parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, choices=['sac', 'td3'], required=True, help="Algorithmus auswählen: sac oder td3")
parser.add_argument('--max_episodes', type=int, default=5000, help="Maximale Episoden für das Training")
parser.add_argument('--max_steps', type=int, default=250, help="Maximale Schritte pro Episode")
parser.add_argument('--batch_size', type=int, default=128, help="Batch Size für das Training")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="Lernrate für den Algorithmus")
parser.add_argument('--gamma', type=float, default=0.99, help="Discount Faktor für zukünftige Belohnungen")
parser.add_argument('--tau', type=float, default=0.005, help="Soft Update Parameter")
parser.add_argument('--evaluate_every', type=int, default=1000, help="Evaluation nach X Episoden")
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# Umgebung initialisieren
env = h_env.HockeyEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Replay Buffer initialisieren
replay_buffer = ReplayBuffer(capacity=1000000)

# Agent basierend auf `--algo` erstellen
if args.algo == "sac":
    agent = SACAgent(state_dim, action_dim, hidden_dim=256, gamma=args.gamma, tau=args.tau,
                     policy_lr=args.learning_rate, q_lr=args.learning_rate, value_lr=args.learning_rate, device=args.device)
elif args.algo == "td3":
    agent = TD3Agent(state_dim, action_dim, hidden_dim=256, gamma=args.gamma, tau=args.tau,
                     policy_lr=args.learning_rate, q_lr=args.learning_rate, device=args.device)

# Training starten
episode_rewards = []
for episode in range(args.max_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(args.max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(replay_buffer) > args.batch_size:
            agent.update(replay_buffer, args.batch_size)

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode+1}: Reward = {episode_reward}")

    # Evaluierung
    if (episode + 1) % args.evaluate_every == 0:
        evaluate_agent(agent, env, num_episodes=10)

# Training abgeschlossen
env.close()
print("Training beendet!")
