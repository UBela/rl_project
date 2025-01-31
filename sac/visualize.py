import numpy as np
import hockey.hockey_env as h_env
import torch
import time
from sac import SACAgent  # Stelle sicher, dass dein Agent in einer separaten Datei ist

# Lade die Hockey-Umgebung
env = h_env.HockeyEnv()
player2 = h_env.BasicOpponent()

# Lade den trainierten SAC-Agenten
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Ersetze "sac_agent.pth" mit deinem gespeicherten Modellpfad
agent = SACAgent(state_dim, action_dim)
agent.policy_net.load_state_dict(torch.load("sac_agent.pth"))
agent.policy_net.eval()

# Anzahl der Spiele, die visualisiert werden sollen
num_episodes = 5

for episode in range(num_episodes):
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    done = False
    episode_reward = 0

    while not done:
        env.render(mode="human")  # Zeigt die GUI

        # SAC-Agent trifft eine Entscheidung
        state = torch.FloatTensor(obs).unsqueeze(0)
        action, _ = agent.policy_net.sample(state)
        action = action.detach().numpy()[0]

        # Gegner trifft eine Entscheidung
        action_opponent = player2.act(obs_agent2)

        # Setze die Aktionen in die Umgebung ein
        obs, reward, done, _, info = env.step(np.hstack([action, action_opponent]))
        obs_agent2 = env.obs_agent_two()
        episode_reward += reward

        # 🔍 Debugging: Detaillierte Infos ausgeben
        print(f"Step Reward: {reward:.2f} | Puck Pos: {info.get('puck_x_position', 'N/A')}, {info.get('puck_y_position', 'N/A')}")
        print(f"Distance to Goal: {info.get('distance_to_goal', 'N/A')} | Puck Speed: {info.get('puck_speed', 'N/A')}")
        print(f"Winner so far: {info.get('winner', 'N/A')}")
        print("------------------------------------------------------")

        time.sleep(0.02)  # Damit die GUI flüssig läuft

    print(f"🏒 Episode {episode + 1} beendet! Gesamt-Reward: {episode_reward:.2f}, Gewinner: {info['winner']}")

env.close()
