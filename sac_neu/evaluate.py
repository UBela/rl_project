import torch
import numpy as np
import hockey.hockey_env as h_env
from sac_agent import SACAgent
import matplotlib.pyplot as plt

# Umgebung
env = h_env.HockeyEnv()
basic_opponent = h_env.BasicOpponent(weak=False)

# Modell laden
agent = SACAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
agent.policy_net.load_state_dict(torch.load("saved_models/sac_model.pth"))
agent.policy_net.eval()

# Testen
NUM_EPISODES = 100
wins, losses, rewards = 0, 0, []

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _ = agent.policy_net.sample(torch.FloatTensor(state))
        action = action.detach().cpu().numpy()
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        state = next_state

    rewards.append(episode_reward)
    if episode_reward > 0:
        wins += 1
    else:
        losses += 1

# Win-Rate berechnen
win_rate = (wins / NUM_EPISODES) * 100
print(f"Win-Rate: {win_rate:.2f}% ({wins} Siege, {losses} Niederlagen)")

# Rewards plotten
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Test Rewards")
plt.axhline(y=0, color="r", linestyle="--", label="Win/Loss Grenze")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SAC vs. BasicOpponent - Test Ergebnisse")
plt.legend()
plt.savefig("plots/sac_evaluation.png")
plt.show()
