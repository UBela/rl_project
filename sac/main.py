import gymnasium as gym
import torch
import numpy as np
from sac import SACAgent, ReplayBuffer

# Hyperparameters
ENV_NAME = "Pendulum-v1"
SEED = 42  # Andere Zufallsseed für bessere Variabilität
BATCH_SIZE = 128  # Größerer Batch für stabileres Training
HIDDEN_DIM = 512  # Größeres Netzwerk für komplexere Aufgaben
REPLAY_BUFFER_SIZE = 500000  # Größerer Replay Buffer für mehr Erfahrung
MAX_FRAMES = 50000  # Längeres Training für bessere Performance
MAX_STEPS = 200  # Gleichbleibend, da dies der Episodenlänge entspricht
EVAL_INTERVAL = 2000  # Selteneres Logging, um Schwankungen zu reduzieren
LEARNING_RATE = 1e-3  # Schnellere Konvergenz (falls instabil, auf 3e-4 reduzieren)
GAMMA = 0.98  # Leicht reduziert für langfristigere Belohnungen
TAU = 0.01  # Langsamere Aktualisierung der Zielnetzwerke
ALPHA = 0.1  # Weniger Exploration für stabileres Lernen


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize SAC Agent
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

    # Training variables
    total_frames = 0
    rewards = []

    while total_frames < MAX_FRAMES:
        state, _ = env.reset()  # Gymnasium liefert ein Tuple zurück
        episode_reward = 0

        for step in range(MAX_STEPS):
            # Select action
            state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
            action, _ = agent.policy_net.sample(state_tensor)
            action = action.detach().cpu().numpy()[0]

            # Step in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_frames += 1

            # Update SAC agent
            if len(replay_buffer) > BATCH_SIZE:
                agent.update(replay_buffer, BATCH_SIZE)

            if done:
                break

        rewards.append(episode_reward)

        # Logging and evaluation
        if total_frames % EVAL_INTERVAL == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Frames: {total_frames}, Avg Reward: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
