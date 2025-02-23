import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import os
import time
import torch
import numpy as np
import json
import copy
from tqdm import tqdm
from evaluate import evaluate
from utils.replay_buffer import PriorityReplayBuffer
from replay_buffer import ReplayBuffer
from hockey import hockey_env as h_env
from contextlib import redirect_stdout	

class SACTrainer:
    """
    SACTrainer ist der Trainer für den SAC-Agenten. Er kümmert sich um das Training,
    das Logging und das Speichern von Modellen.
    """

    def __init__(self, logger, config):
        self.logger = logger
        self._config = config
        self.total_steps = 0
        self.total_gradient_steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Replay Buffer initialisieren (mit oder ohne Prioritized Experience Replay)
        if config["use_PER"]:
            self.replay_buffer = PriorityReplayBuffer(config["buffer_size"], alpha=config["per_alpha"], beta=config["per_beta"])
        else:
            self.replay_buffer = ReplayBuffer(config["buffer_size"])

        # Log-Datei für Ergebnisse vorbereiten
        self.log_results_filename = f"results/{self._config['results_folder']}/evaluation_log.json"
        os.makedirs(os.path.dirname(self.log_results_filename), exist_ok=True)
        if not os.path.exists(self.log_results_filename):
            with open(self.log_results_filename, "w") as f:
                json.dump([], f)

    def _select_opponent(self, opponents, i_episode, win_rate):
        """
        Wählt den Gegner basierend auf der Episode und Winrate aus.
        """
        phase1 = int(self._config["max_episodes"] * 0.3)  # Erste 30%: Nur schwache Gegner
        phase2 = int(self._config["max_episodes"] * 0.6)  # Bis 60%: Mischung
        phase3 = int(self._config["max_episodes"] * 0.8)  # Danach Self-Play
        
        if i_episode < phase1:
            return opponents[0]  # Nur schwache Gegner
        elif i_episode < phase2:
            return np.random.choice([opponents[0], opponents[1]])  # Mischung
        elif win_rate < 0.3 and i_episode < phase3:
            return opponents[1]  # Stärkerer Gegner nur wenn nötig
        else:
            return np.random.choice(opponents)  # Volles Self-Play aktiv

    def _add_self_play_agent(self, agent, opponents, i_episode):
        """
        Fügt das aktuelle Modell als Self-Play-Gegner hinzu.
        """
        if not self._config['use_self_play']:
            return
        if i_episode >= self._config['self_play_start'] and self.total_gradient_steps % self._config['self_play_intervall'] == 0:
            print("Adding agent to self-play opponents...")
            opponents.append(copy.deepcopy(agent))

    def fill_replay_buffer(self, agent, env):
        """
        Befüllt den Replay Buffer mit zufälligen Aktionen, um eine initiale Trainingsbasis zu schaffen,
        während störende print-Ausgaben aus der hockey_env unterdrückt werden.
        """
        print("⏳ Filling replay buffer with random actions...")

        with open(os.devnull, 'w') as f, redirect_stdout(f):  # 🛑 Unterdrückt alle Prints
            with tqdm(total=self._config["buffer_size"], desc="⏳ Filling Replay Buffer", unit="samples") as pbar:
                while len(self.replay_buffer) < self._config["buffer_size"]:
                    ob, _ = env.reset()
                    obs_agent2 = env.obs_agent_two()
                    opponent = self._select_opponent([h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)], i_episode=0, win_rate=0.0)
                    done, trunc = False, False

                    while not (done or trunc):
                        a1 = np.random.uniform(-1, 1, env.action_space.shape[0] // 2)
                        a2 = opponent.act(obs_agent2)
                        actions = np.hstack([a1, a2])
                        (ob_new, reward, done, trunc, _info) = env.step(actions)

                        self.replay_buffer.add_transition([ob, a1, reward, ob_new, done])
                        pbar.update(1)
                        ob = ob_new
                        obs_agent2 = env.obs_agent_two()

        print(f"✅ Replay buffer filled with {len(self.replay_buffer)} samples.")

    def train(self, agent, opponents, env):
        """
        Haupttrainingsloop für den SAC-Agenten.
        """
        rew_stats, q1_losses, q2_losses, actor_losses, alpha_losses = [], [], [], [], []
        eval_stats = {"weak": {"reward": [], "touch": [], "won": [], "lost": []},
                      "strong": {"reward": [], "touch": [], "won": [], "lost": []}}
        episode_counter, total_step_counter, grad_updates = 1, 0, 0

        # Fülle den Replay Buffer zu Beginn
        self.fill_replay_buffer(agent, env)

        while episode_counter <= self._config['max_episodes']:
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            total_reward, touched = 0, 0

            opponent = self._select_opponent(opponents, episode_counter, 0.0)  # Winrate wird später aktualisiert
            first_time_touch = 1

            for step in range(self._config['max_timesteps']):
                a1 = agent.select_action(ob)
                a2 = opponent.act(obs_agent2)
                actions = np.hstack([a1, a2])
                next_state, reward, done, truncated, _info = env.step(actions)

                touched = max(touched, _info.get('reward_touch_puck', 0.0))

                step_reward = (
                    reward
                    + 5 * _info.get('reward_closeness_to_puck', 0.0)
                    - (1 - touched) * 0.1
                    + touched * first_time_touch * 0.1 * step
                )
                first_time_touch = 1 - touched
                total_reward += step_reward

                agent.replay_buffer.add_transition((ob, a1, step_reward, next_state, done))

                ob = next_state
                obs_agent2 = env.obs_agent_two()
                total_step_counter += 1

                if done or truncated:
                    break

            # Training des SAC-Agenten
            if len(agent.replay_buffer) >= self._config['batch_size']:
                for _ in range(self._config['grad_steps']):
                    losses = agent.update(agent.replay_buffer, self._config['batch_size'])

                    grad_updates += 1
                    q1_losses.append(losses[0])
                    q2_losses.append(losses[1])
                    actor_losses.append(losses[2])
                    alpha_losses.append(losses[3])

            # Lernraten-Update
            agent.schedulers_step()
            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward)
            avg_reward = np.mean(rew_stats[-100:])  # Durchschnittlicher Reward der letzten 100 Episoden
            print(f"Episode {episode_counter}: Reward={total_reward:.3f}, Avg. Reward (100 Episoden)={avg_reward:.3f}")

            # Evaluierung
            if episode_counter % self._config['evaluate_every'] == 0:
                agent.eval()
                for eval_op in ['strong', 'weak']:
                    ev_opponent = opponents[0] if eval_op == 'strong' else h_env.BasicOpponent(False)
                    with open(os.devnull, 'w') as f, redirect_stdout(f):  
                        rew, touch, won, lost = evaluate(agent, env, ev_opponent, 100)
                    eval_stats[eval_op]['reward'].append(rew)
                    eval_stats[eval_op]['touch'].append(touch)
                    eval_stats[eval_op]['won'].append(won)
                    eval_stats[eval_op]['lost'].append(lost)
                    win_rate = sum(won) / 100
                    loss_rate = sum(lost) / 100
                    opponent_name = "STRONG OPPONENT" if eval_op == "strong" else "WEAK OPPONENT"
                    print(f"Evaluation {opponent_name}:")
                    print(f"Win Rate: {win_rate:.3f} | Loss Rate: {loss_rate:.3f}")

                self.logger.save_model(agent, f'checkpoint_{episode_counter}')
                agent.train()

            rew_stats.append(total_reward)
            episode_counter += 1

        # Finale Speicherung & Logging
        self.logger.print_stats(rew_stats, {}, {}, {})  # Kein loser Tracking
        self.logger.save_model(agent, 'final_agent.pkl')
        self.logger.plot_running_mean(rew_stats, "Total Reward", "reward.pdf")

        print("✅ Training abgeschlossen!")
