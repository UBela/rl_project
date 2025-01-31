import numpy as np
import torch
import hockey.hockey_env as h_env
from sac_agent import SACAgent  # Importiere euren SAC-Agenten
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_shooting(agent, episodes=1000, save_path="sac_shooting.pth"):
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action, _ = agent.policy_net.sample(torch.FloatTensor(state).to(device))
            action = action.cpu().detach().numpy()

            # Gegner bleibt passiv
            opponent_action = np.zeros(4)
            full_action = np.hstack([action, opponent_action])

            next_state, reward, done, truncated, info = env.step(full_action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(agent.replay_buffer) > 64:
                agent.update(agent.replay_buffer, batch_size=64)

        if episode % 100 == 0:
            print(f"[SHOOTING] Episode {episode}, Reward: {reward}")

    torch.save(agent.policy_net.state_dict(), save_path)
    env.close()

def train_defense(agent, episodes=1000, save_path="sac_defense.pth"):
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)

    for episode in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action, _ = agent.policy_net.sample(torch.FloatTensor(state).to(device))
            action = action.cpu().detach().numpy()

            # Gegner bleibt passiv
            opponent_action = np.zeros(4)
            full_action = np.hstack([action, opponent_action])

            next_state, reward, done, truncated, info = env.step(full_action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(agent.replay_buffer) > 64:
                agent.update(agent.replay_buffer, batch_size=64)

        if episode % 100 == 0:
            print(f"[DEFENSE] Episode {episode}, Reward: {reward}")

    torch.save(agent.policy_net.state_dict(), save_path)
    env.close()

if __name__ == "__main__":
    state_dim = 24  # Je nach Umgebung anpassen
    action_dim = 4
    agent = SACAgent(state_dim, action_dim, device=device)

    # Shooting Training
    train_shooting(agent, episodes=1000)

    # Defense Training
    train_defense(agent, episodes=1000)

    print("Shooting und Defense Training abgeschlossen. Agent gespeichert.")
