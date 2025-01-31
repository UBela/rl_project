import gymnasium as gym
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sac import SACAgent, ReplayBuffer
import hockey.hockey_env as h_env

# Hyperparameter
ENV_NAME = "Hockey"
SEED = 123
BATCH_SIZE = 64
HIDDEN_DIM = 256
REPLAY_BUFFER_SIZE = 100000
MAX_FRAMES = 300000  # Längeres Training
MAX_STEPS = 250
EVAL_INTERVAL = 10000  # Seltener loggen
LEARNING_RATE = 1e-4  # Reduzierte Lernrate
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
SELF_PLAY_INTERVAL = 50000  # Seltenerer Wechsel

# Setze den Seed für Reproduzierbarkeit
np.random.seed(SEED)
torch.manual_seed(SEED)

# Umgebung erstellen
env = h_env.HockeyEnv()
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

# Agent initialisieren
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=HIDDEN_DIM,
    gamma=GAMMA,
    tau=TAU,
    alpha=ALPHA,
    policy_lr=LEARNING_RATE,
    q_lr=LEARNING_RATE,
    value_lr=LEARNING_RATE,
    device=device,
)

# Replay Buffer
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# Gegner-Konfiguration (erst leichter Gegner, dann steigern)
weak_opponent = h_env.BasicOpponent(weak=True)
basic_opponent = h_env.BasicOpponent(weak=False)
opponent = weak_opponent  # Start mit einfachem Gegner
self_play_active = False
self_play_agent = None

# Trainingsvariablen
total_frames = 0
rewards = []
opponent_change_counter = 0
agent_side = "Left"

print("🚀 Starte Training...")

previous_frames = total_frames  # Initialisierung vor der Episode

# Funktion zum Speichern des Modells
def save_model(agent, frame, filename="sac_agent.pth"):
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", filename)
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"💾 Modell gespeichert: {save_path}")

# 📌 PHASE 1: Shooting Training
def train_shooting(agent, episodes=1000):
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action, _ = agent.policy_net.sample(torch.FloatTensor(state).to(device))
            action = action.cpu().detach().numpy()

            opponent_action = np.zeros(4)  # Gegner bleibt passiv
            full_action = np.hstack([action, opponent_action])

            next_state, reward, done, _, _ = env.step(full_action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > BATCH_SIZE:
                agent.update(replay_buffer, batch_size=BATCH_SIZE)

        if episode % 100 == 0:
            print(f"[SHOOTING] Episode {episode}, Reward: {reward}")

    save_model(agent, total_frames, filename="sac_shooting.pth")
    env.close()

# 📌 PHASE 2: Defense Training
def train_defense(agent, episodes=1000):
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action, _ = agent.policy_net.sample(torch.FloatTensor(state).to(device))
            action = action.cpu().detach().numpy()

            opponent_action = np.zeros(4)  # Gegner bleibt passiv
            full_action = np.hstack([action, opponent_action])

            next_state, reward, done, _, _ = env.step(full_action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > BATCH_SIZE:
                agent.update(replay_buffer, batch_size=BATCH_SIZE)

        if episode % 100 == 0:
            print(f"[DEFENSE] Episode {episode}, Reward: {reward}")

    save_model(agent, total_frames, filename="sac_defense.pth")
    env.close()

# 📌 PHASE 3: Normales Training mit Self-Play
while total_frames < MAX_FRAMES:
    state, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        reward = np.clip(reward / (1 + abs(reward)), -1, 1)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        total_frames += 1

        # Self-Play Wechsel prüfen
        if total_frames // SELF_PLAY_INTERVAL > previous_frames // SELF_PLAY_INTERVAL:
            self_play_active = not self_play_active
            opponent = agent if self_play_active else basic_opponent
            print(f"🔄 Gegner gewechselt: {'Self-Play' if self_play_active else 'BasicOpponent'}")

        if total_frames // EVAL_INTERVAL > previous_frames // EVAL_INTERVAL:
            avg_reward = np.mean(rewards[-10:])
            print(f"📊 Frames: {total_frames}, Avg. Reward: {avg_reward:.2f}")
            save_model(agent, total_frames)

        if done:
            break

    rewards.append(episode_reward)
    previous_frames = total_frames

# 📌 PHASE 4: Test gegen Basic Opponent & Visualisierung
def test_agent(agent, episodes=100):
    env = h_env.HockeyEnv()
    opponent = h_env.BasicOpponent()
    test_rewards = []
    wins = 0

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = agent.policy_net.sample(torch.FloatTensor(state).to(device))
            action = action.cpu().detach().numpy()

            opponent_action = opponent.act(state)
            full_action = np.hstack([action, opponent_action])

            next_state, reward, done, _, info = env.step(full_action)
            total_reward += reward
            state = next_state

        test_rewards.append(total_reward)
        if info["winner"] == 1:
            wins += 1

    win_rate = wins / episodes
    print(f"Win-Rate gegen Basic Opponent: {win_rate * 100:.2f}%")
    env.close()

    # Plots generieren
    plt.figure()
    plt.plot(test_rewards, label="Test Rewards")
    plt.axhline(y=0, color="r", linestyle="--", label="Win/Loss Grenze")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SAC vs. BasicOpponent - Test Ergebnisse")
    plt.legend()
    plt.savefig("test_rewards.png")
    plt.show()

# **Test durchführen**
test_agent(agent, episodes=100)

env.close()
print("✅ Training abgeschlossen!")
