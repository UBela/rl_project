import torch
import numpy as np
import argparse
from hockey import hockey_env as h_env
from sac_agent import SACAgent

# Parameter für den Test
NUM_GAMES = 100  # Anzahl der Test-Spiele

def load_agent(pth_path, state_dim, action_space, config):
    """Lädt einen SAC-Agenten aus einer .pth-Datei."""
    agent = SACAgent(state_dim, action_space, config)

    # 📥 Speichermodell laden
    state = torch.load(pth_path, map_location=torch.device('cpu'))

    # 🔄 Richtige Schlüssel extrahieren
    agent.policy_net.load_state_dict(state["policy_net"])
    agent.qnet1.load_state_dict(state["qnet1"])
    agent.qnet_target.load_state_dict(state["qnet_target"])

    agent.policy_net.eval()
    return agent


def test_agent(agent, env, opponent, num_games=NUM_GAMES):
    """Lässt den Agenten gegen einen Gegner antreten und berechnet die Win-Rate."""
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        state, _ = env.reset()
        done = False

        while not done:
            action_agent = agent.act(state)
            action_opponent = opponent.act(env.obs_agent_two())

            actions = np.hstack([action_agent, action_opponent])
            state, reward, done, _, info = env.step(actions)

        # Gewinner evaluieren
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
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model_path", type=str, required=True, help="Pfad zur .pth-Datei des Modells")
    #parser.add_argument("--mode", type=str, default="normal", help="Trainingsmodus (shooting | defense | normal)")
    #args = parser.parse_args()

    # Umgebung erstellen
    mode_mapping = {
        "normal": h_env.Mode.NORMAL,
        "shooting": h_env.Mode.TRAIN_SHOOTING,
        "defense": h_env.Mode.TRAIN_DEFENSE
    }

    #if args.mode not in mode_mapping:
    #    raise ValueError("Ungültiger Modus! Verwende: shooting | defense | normal")

    env = h_env.HockeyEnv(mode=mode_mapping["normal"])

    # Agentenparameter aus Environment bestimmen
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space

    # Standardkonfiguration für den Agenten (muss an dein Training angepasst werden!)
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
    
    # Prioritized Experience Replay (PER)
    "use_PER": True,
    "per_alpha": 0.6,
    "per_beta": 0.4,
    "per_beta_update": None,

    # SAC Hyperparameter
    "policy_lr": 0.001,
    "q_lr": 0.001,
    "value_lr": 0.001,
    "tau": 0.005,
    "gamma": 0.95,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "alpha_lr": 0.0001,

    # Training Settings
    "batch_size": 128,
    "buffer_size": 131072,
    "cuda": False,
    "show": False,
    "q": False,
    "evaluate": False,
    "mode": "normal",
    "preload_path": None,
    "transitions_path": None,
    "add_self_every": 100000,

    # Lernraten-Anpassung
    "lr_factor": 0.5,
    "lr_milestones": [10000, 18000],
    "alpha_milestones": [10000, 18000],
    "update_target_every": 1,
    "grad_steps": 32,
    "soft_tau": 0.005
}


    # Agent laden
    print(f"📥 Lade Agent aus 19570.pth")
    agent = load_agent(r"C:\Users\regin\Documents\Winter24_25\rl_project\sac\logs\agents\500.pth", state_dim, action_space, config)
    print("✅ Modell erfolgreich geladen!")

    # Gegner erstellen
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False)

    # Test gegen schwachen Gegner
    print("\n🟢 Test gegen **schwachen** Gegner ...")
    win_rate_weak, loss_rate_weak, draw_rate_weak = test_agent(agent, env, weak_opponent)
    print(f"➡ Win-Rate: {win_rate_weak:.2f}% | Loss-Rate: {loss_rate_weak:.2f}% | Draw-Rate: {draw_rate_weak:.2f}%")

    # Test gegen starken Gegner
    print("\n🔴 Test gegen **starken** Gegner ...")
    win_rate_strong, loss_rate_strong, draw_rate_strong = test_agent(agent, env, strong_opponent)
    print(f"➡ Win-Rate: {win_rate_strong:.2f}% | Loss-Rate: {loss_rate_strong:.2f}% | Draw-Rate: {draw_rate_strong:.2f}%")

    print("\n🎯 **Test abgeschlossen!**")
