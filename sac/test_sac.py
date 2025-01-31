import gymnasium as gym
import torch
import numpy as np
import hockey.hockey_env as h_env
from sac import SACAgent

# Umgebung erstellen
env = h_env.HockeyEnv()
basic_opponent = h_env.BasicOpponent(weak=False)

# Parameter für das Modell
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modell laden
agent = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=DEVICE)
agent.policy_net.load_state_dict(torch.load("sac_agent.pth"))
agent.policy_net.eval()
print("✅ Modell geladen und bereit zum Testen!")

# Teste das Modell gegen BasicOpponent
NUM_EPISODES = 100  # Anzahl der Testspiele
wins = 0
losses = 0
rewards = []

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _ = agent.policy_net.sample(torch.FloatTensor(state).to(DEVICE))
        action = action.cpu().detach().numpy()
        next_state, reward, done, _, _ = env.step(action)

        episode_reward += reward
        state = next_state

    rewards.append(episode_reward)
    
    if episode_reward > 0:  # Annahme: Positiver Reward = Win
        wins += 1
    else:
        losses += 1

    print(f"Spiel {episode+1}/{NUM_EPISODES} - Reward: {episode_reward:.2f}")

# Ergebnisse ausgeben
win_rate = wins / NUM_EPISODES * 100
print(f"🎯 Win-Rate: {win_rate:.2f}% ({wins} Siege, {losses} Niederlagen)")

# Plot der Rewards
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Test Rewards")
plt.axhline(y=0, color="r", linestyle="--", label="Win/Loss Grenze")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SAC vs. BasicOpponent - Test Ergebnisse")
plt.legend()
plt.savefig("test_rewards.png")
plt.show()

env.close()
