import gymnasium as gym
import torch
import numpy as np
from sac import SACAgent, ReplayBuffer

ENV_NAME = "Pendulum-v1"
SEED = 42
BATCH_SIZE = 128
HIDDEN_DIM = 512
REPLAY_BUFFER_SIZE = 1000000
MAX_FRAMES = 300000
MAX_STEPS = 200
EVAL_INTERVAL = 2000
LEARNING_RATE = 3e-4
GAMMA = 0.98
TAU = 0.005
ALPHA = 0.1

def evaluate_agent(env, agent, episodes=10):
    total_reward = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, _ = agent.policy_net.sample(state_tensor)
            action = action.detach().cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
    return total_reward / episodes


def main():
    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

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
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    total_frames = 0
    rewards = []

    while total_frames < MAX_FRAMES:
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, _ = agent.policy_net.sample(state_tensor)
            action = action.detach().cpu().numpy()[0]

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_frames += 1

            if len(replay_buffer) > BATCH_SIZE:
                agent.update(replay_buffer, BATCH_SIZE)

            if done:
                break

        rewards.append(episode_reward)

        if total_frames % EVAL_INTERVAL == 0:
            avg_reward = evaluate_agent(env, agent)
            print(f"Frames: {total_frames}, Avg Reward: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
