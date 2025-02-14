import gymnasium as gym
import torch
import numpy as np
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
MAX_FRAMES = 5000000  # Verlängertes Training
MAX_STEPS = 250
EVAL_INTERVAL = 10000  # Log-Intervall
LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
SELF_PLAY_INTERVAL = 50000  # Self-Play-Intervall

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

# Gegner-Konfiguration
opponent = h_env.BasicOpponent(weak=False)  # Start gegen BasicOpponent
self_play_active = False

# Speicherung der Rewards
total_frames = 0
rewards = []
closeness_rewards = []
touch_rewards = []
puck_direction_rewards = []

print("🚀 Starte Training...")

# Funktion zum Speichern des Modells
def save_model(agent, frame):
    save_path = f"models/sac_agent_{frame}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"💾 Modell gespeichert: {save_path}")

while total_frames < MAX_FRAMES:
    state, _ = env.reset()
    obs_agent2 = env.obs_agent_two()
    episode_reward = 0
    episode_closeness = 0
    episode_touch = 0
    episode_puck_direction = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        action_opponent = opponent.act(obs_agent2)

        next_state, reward, done, _, info = env.step(np.hstack([action, action_opponent]))
        obs_agent2 = env.obs_agent_two()

        # Speichere die einzelnen Reward-Komponenten
        episode_closeness += info["reward_closeness_to_puck"]
        episode_touch += info["reward_touch_puck"]
        episode_puck_direction += info["reward_puck_direction"]

        # Gesamt-Reward speichern
        episode_reward += reward

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_frames += 1

        # Logging nach jedem EVAL_INTERVAL
        if total_frames % EVAL_INTERVAL == 0:
            avg_reward = np.mean(rewards[-10:]) if len(rewards) > 10 else 0
            print(
                f"📊 Frames: {total_frames}, Avg. Reward: {avg_reward:.2f}, "
                f"Opponent: {'Self-Play' if self_play_active else type(opponent).__name__}"
            )
            save_model(agent, total_frames)

        if done:
            break

    # Episode-Daten speichern
    rewards.append(episode_reward)
    closeness_rewards.append(episode_closeness)
    touch_rewards.append(episode_touch)
    puck_direction_rewards.append(episode_puck_direction)

# 📈 **Visualisierung der Ergebnisse**
plt.figure(figsize=(12, 6))

# **Plot Gesamt-Reward**
plt.subplot(2, 2, 1)
plt.plot(rewards, label="Gesamt-Reward", alpha=0.3)
plt.plot(np.convolve(rewards, np.ones(100) / 100, mode="valid"), label="Moving Average (100)", color="blue")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress - Gesamt")
plt.legend()

# **Plot Reward für Closeness to Puck**
plt.subplot(2, 2, 2)
plt.plot(closeness_rewards, label="Closeness to Puck", alpha=0.3, color="orange")
plt.plot(np.convolve(closeness_rewards, np.ones(100) / 100, mode="valid"), label="Moving Average (100)", color="red")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Closeness to Puck Reward")
plt.legend()

# **Plot Reward für Puck Touch**
plt.subplot(2, 2, 3)
plt.plot(touch_rewards, label="Touch Puck", alpha=0.3, color="green")
plt.plot(np.convolve(touch_rewards, np.ones(100) / 100, mode="valid"), label="Moving Average (100)", color="darkgreen")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Touch Puck Reward")
plt.legend()

# **Plot Reward für Puck Direction**
plt.subplot(2, 2, 4)
plt.plot(puck_direction_rewards, label="Puck Direction", alpha=0.3, color="purple")
plt.plot(np.convolve(puck_direction_rewards, np.ones(100) / 100, mode="valid"), label="Moving Average (100)", color="darkviolet")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Puck Direction Reward")
plt.legend()

plt.tight_layout()
plt.savefig("training_rewards_detailed.png")
plt.show()

env.close()
print("✅ Training abgeschlossen!")
