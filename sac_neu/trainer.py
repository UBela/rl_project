import numpy as np
import torch
import os
import json
from hockey.hockey_env import BasicOpponent

class Trainer:
    def __init__(self, agent, env, replay_buffer, args):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.args = args
        self.opponent = BasicOpponent(weak=True)  # Start mit einfachem Gegner
        self.current_phase = "random"  # Start mit Random Exploration
        self.save_dir = "saved_models"
        self.logs_file = "training_logs.json"

        os.makedirs(self.save_dir, exist_ok=True)
        self.training_logs = []

        # **Prozentuale Grenzen für die Trainingsphasen**
        self.phase_limits = {
            "random": 0.2,        # 20% zufällige Aktionen
            "weak_opponent": 0.4, # 20% gegen schwachen Gegner
            "self_play": 0.7,     # 30% Self-Play
            "best_sac": 1.0       # 30% gegen bestes gespeichertes Modell
        }

    def train(self):
        for episode in range(self.args.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            # ** Phase 1: Zufällige Aktionen für Exploration**
            if episode / self.args.max_episodes < self.phase_limits["random"]:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(state)

            for step in range(self.args.max_steps):
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.add_transition([state, action, reward, next_state, done])
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > self.args.batch_size:
                    self.agent.update(self.replay_buffer, self.args.batch_size)

                if done:
                    break

            # **Dynamischer Wechsel der Trainingsstrategie**
            self.update_training_strategy(episode)

            # ** Modell speichern & Logging**
            if (episode + 1) % 5 == 0:
                save_path = os.path.join(self.save_dir, f"sac_model_{episode+1}.pth")
                torch.save(self.agent.policy_net.state_dict(), save_path)
                print(f"✅ Modell gespeichert unter: {save_path}")
                self.training_logs.append({
                    "episode": episode + 1,
                    "reward": episode_reward
                })
                with open(self.logs_file, "w") as f:
                    json.dump(self.training_logs, f)

            # ** Evaluierung**
            if (episode + 1) % self.args.evaluate_every == 0:
                win_rate = self.evaluate()
                print(f"🎯 Win-Rate nach {episode+1} Episoden: {win_rate:.2f}%")

    def update_training_strategy(self, episode):
        """
        Dynamische Strategie-Anpassung basierend auf Prozentanteilen der max_episodes.
        """
        progress = episode / self.args.max_episodes

        if progress < self.phase_limits["random"]:
            new_phase = "random"
        elif progress < self.phase_limits["weak_opponent"]:
            new_phase = "weak_opponent"
            self.opponent = BasicOpponent(weak=True)
        elif progress < self.phase_limits["self_play"]:
            new_phase = "self_play"
            self.opponent = self.agent  # Self-Play
        else:
            new_phase = "best_sac"
            best_model = max(os.listdir(self.save_dir), key=lambda f: int(f.split("_")[-1].split(".")[0]))
            best_model_path = os.path.join(self.save_dir, best_model)
            self.agent.policy_net.load_state_dict(torch.load(best_model_path))
            self.agent.policy_net.eval()

        if new_phase != self.current_phase:
            print(f"🔄 Wechsel zu {new_phase}!")
            self.current_phase = new_phase

    def evaluate(self, num_episodes=10):
        """
        Testet das aktuelle Modell über mehrere Episoden und gibt die Win-Rate zurück.
        """
        wins, losses = 0, 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            if episode_reward > 0:
                wins += 1
            else:
                losses += 1

        return (wins / num_episodes) * 100
