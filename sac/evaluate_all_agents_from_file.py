import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  
from hockey import hockey_env as h_env
from sac_agent import SACAgent


NUM_GAMES = 100 
SAVE_PLOT_PATH = "./logs/evaluation.png"  


def load_agent(pth_path, state_dim, action_space, config):
    """Loads an SAC agent from a .pth file."""
    agent = SACAgent(state_dim, action_space, config)


    state = torch.load(pth_path, map_location=torch.device('cpu'))
    
    agent.policy_net.load_state_dict(state["policy_net"])
    agent.qnet1.load_state_dict(state["qnet1"])
    agent.qnet_target.load_state_dict(state["qnet_target"])

    agent.policy_net.eval()
    return agent


def test_agent(agent, env, opponent, num_games=NUM_GAMES):
    """Plays multiple games and returns win, loss, and draw rates."""
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        state, _ = env.reset()
        done = False

        while not done:
            action_agent = agent.act(state)
            action_opponent = opponent.act(env.obs_agent_two())

            actions = np.hstack([action_agent, action_opponent])
            state, reward, done, _, info = env.step(actions)

        if info["winner"] == 1:
            wins += 1
        elif info["winner"] == -1:
            losses += 1
        else:
            draws += 1

    win_rate = wins / num_games * 100
    loss_rate = losses / num_games * 100
    draw_rate = draws / num_games * 100

    return win_rate, loss_rate, draw_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAC agent models.")
    parser.add_argument("--model_dir", type=str, default="./logs/agents", help="Path to saved models")
    parser.add_argument("--skip", type=int, default=10, help="Evaluate every Nth model (default: 10)")
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    SKIP_NTH = args.skip

    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)

    state_dim = env.observation_space.shape[0]
    action_space = env.action_space

    config = {
    "max_episodes": 40000,
    "max_timesteps": 250,
    "iter_fit": 32,
    "log_interval": 10,
    "random_seed": None,
    "render": False,
    "use_hard_opp": False,
    "selfplay": True,
    "self_play_intervall": 50000,
    "self_play_start": 16000,
    "evaluate_every": 1000,
    "results_folder": "results",
    "hidden_dim": [256, 256],
    
    "use_PER": True,
    "per_alpha": 0.3,
    "per_beta": 0.4,
    "per_beta_update": 0.0006,

    "policy_lr": 0.001,
    "q_lr": 0.001,
    "value_lr": 0.001,
    "tau": 0.005,
    "gamma": 0.95,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "alpha_lr": 0.0001,

    "batch_size": 128,
    "buffer_size": 131072,
    "cuda": True,
    "show": False,
    "q": False,
    "evaluate": False,
    "mode": "defense",
    "preload_path": None,
    "transitions_path": None,
    "add_self_every": 100000,

    "lr_factor": 0.5,
    "lr_milestones": [10000, 18000],
    "alpha_milestones": [10000, 18000],
    "update_target_every": 1,
    "grad_steps": 32,
    "soft_tau": 0.005
    }

    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False)

    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found!")
        exit(1)

    model_files = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")], 
        key=lambda x: int(x.split('.')[0])
    )

    filtered_model_files = [f for f in model_files if int(f.split('.')[0]) % SKIP_NTH == 0]

    if not filtered_model_files:
        print(f"No model files matching the skip condition found in '{MODEL_DIR}'!")
        exit(1)

    episodes = []
    win_rates_weak, loss_rates_weak, draw_rates_weak = [], [], []
    win_rates_strong, loss_rates_strong, draw_rates_strong = [], [], []

    print(f"Evaluating {len(filtered_model_files)} models...")


    for model_file in tqdm(filtered_model_files, desc="Evaluating Models", unit="model"):
        model_path = os.path.join(MODEL_DIR, model_file)
        episode_number = int(model_file.split('.')[0])  

        agent = load_agent(model_path, state_dim, action_space, config)

        win_weak, loss_weak, draw_weak = test_agent(agent, env, weak_opponent)
        win_rates_weak.append(win_weak)
        loss_rates_weak.append(loss_weak)
        draw_rates_weak.append(draw_weak)

        win_strong, loss_strong, draw_strong = test_agent(agent, env, strong_opponent)
        win_rates_strong.append(win_strong)
        loss_rates_strong.append(loss_strong)
        draw_rates_strong.append(draw_strong)

        episodes.append(episode_number)

    print("Evaluation completed. Plotting results...")

    plt.figure(figsize=(12, 6))

    plt.plot(episodes, win_rates_weak, label="Win % (Weak)", linestyle="-", color="green")
    plt.plot(episodes, loss_rates_weak, label="Loss % (Weak)", linestyle="--", color="red")
    plt.plot(episodes, draw_rates_weak, label="Draw % (Weak)", linestyle=":", color="blue")

    plt.plot(episodes, win_rates_strong, label="Win % (Strong)", linestyle="-", color="darkgreen")
    plt.plot(episodes, loss_rates_strong, label="Loss % (Strong)", linestyle="--", color="darkred")
    plt.plot(episodes, draw_rates_strong, label="Draw % (Strong)", linestyle=":", color="darkblue")

    plt.xlabel("Training Step")
    plt.ylabel("Percentage")
    plt.title("Win/Loss/Draw Rates Over Training")
    plt.legend()
    plt.grid(True)

    plt.savefig(SAVE_PLOT_PATH)
    plt.show()

    print(f"Plot saved at: {SAVE_PLOT_PATH}")
