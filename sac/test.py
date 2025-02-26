import torch
import pickle
import numpy as np
from hockey import hockey_env as h_env
from sac_agent import SACAgent

# === CONFIGURATION ===
MODEL_PATH_PTH = r"C:\Users\regin\Documents\Winter24_25\RL_EVAL\test\agent.pkl.pth"  # PTH file path
MODEL_PATH_PKL = r"C:\Users\regin\Documents\Winter24_25\RL_EVAL\test\agent.pkl.pkl"  # PKL file path # Falls `pickle.dump(agent, f)`
TEST_EPISODES = 100  # Number of test episodes
RENDER = False  # Set True if you want to visualize

# === LOAD ENVIRONMENT ===
env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL, verbose=False)
weak_opponent = h_env.BasicOpponent(weak=True)
strong_opponent = h_env.BasicOpponent(weak=False)
config= {'hidden_dim': 256, 'max_episodes': 40000, 'max_timesteps': 250, 'iter_fit': 32, 'log_interval': 20, 'random_seed': None, 'render': False, 'use_hard_opp': False, 'selfplay': False, 'self_play_intervall': 50000, 'self_play_start': 16000, 'evaluate_every': 1000, 'results_folder': 'results', 'gamma': 0.99, 'tau': 0.005, 'alpha': 0.2, 'automatic_entropy_tuning': True, 'policy_lr': 0.0003, 'q_lr': 0.0003, 'value_lr': 0.001, 'alpha_lr': 10000.0, 'lr_milestones': [10000, 18000], 'buffer_size': 131072, 'per_alpha': 0.6, 'per_beta': 0.4, 'per_beta_update': None, 'use_PER': True, 'alpha_milestones': [10000, 18000], 'target_update_interval': 1, 'soft_tau': 0.005, 'batch_size': 128, 'cuda': False, 'show': False, 'q': False, 'evaluate': False, 'mode': 'normal', 'transitions_path': None, 'add_self_every': 100000, 'lr_factor': 0.5, 'update_target_every': 1, 'grad_steps': 32}


def load_agent():
    """Load the trained agent from PTH or PKL file."""
    try:
        # Try loading full agent from PKL
        with open(MODEL_PATH_PKL, "rb") as f:
            agent = pickle.load(f)
        print(f"✅ Loaded agent from {MODEL_PATH_PKL} (CPU mode)")
        return agent

    except Exception:
        try:
            # Try loading state_dict from PTH
            state_dict = torch.load(MODEL_PATH_PTH, map_location=torch.device("cpu"))
            print(f"✅ Loaded state_dict from {MODEL_PATH_PTH}")

            # Re-create the SACAgent and load parameters
            agent = SACAgent(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dim=256,  # This should match what was used during training
                config=config
            )
            #agent.load_state_dict(state_dict)
            #agent.eval()
            return agent

        except Exception as e:
            print(f"❌ Failed to load agent: {e}")
            exit(1)


def test_agent(agent, opponent, episodes=100):
    """Evaluate the agent against a given opponent."""
    wins, losses, ties = 0, 0, 0
    total_reward = []

    for ep in range(episodes):
        ob, _ = env.reset()
        obs_agent2 = env.obs_agent_two()
        done = False
        ep_reward = 0

        while not done:
            action_agent = agent.act(ob)
            action_opponent = opponent.act(obs_agent2)
            actions = np.hstack([action_agent, action_opponent])

            ob_new, reward, done, trunc, info = env.step(actions)
            ep_reward += reward

            if RENDER:
                env.render()

            ob = ob_new
            obs_agent2 = env.obs_agent_two()

        winner = info.get("winner", 0)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            ties += 1
        total_reward.append(ep_reward)

    win_rate = wins / episodes
    loss_rate = losses / episodes
    tie_rate = ties / episodes
    avg_reward = np.mean(total_reward)

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "tie_rate": tie_rate,
        "avg_reward": avg_reward,
    }


if __name__ == "__main__":
    agent = load_agent()

    print("\n=== Testing against WEAK OPPONENT ===")
    weak_results = test_agent(agent, weak_opponent, TEST_EPISODES)
    print(weak_results)

    print("\n=== Testing against STRONG OPPONENT ===")
    strong_results = test_agent(agent, strong_opponent, TEST_EPISODES)
    print(strong_results)

    env.close()
