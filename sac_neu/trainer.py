import numpy as np
import torch
import os
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

        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        for episode in range(self.args.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            # **🎲 Phase 1: Zufällige Aktionen für Exploration**
            if episode < 10000:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(state)

            for step in range(self.args.max_steps):
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.add_transition([state, action, reward, next_state, done])
                state = next_state
                episode_reward += reward

                # Erst nach Auffüllen des Replay Buffers trainieren
                if len(self.replay_buffer) > self.args.batch_size:
                    self.agent.update(self.replay_buffer, self.args.batch_size)

                if done:
                    break

            print(f"Episode {episode+1}: Reward = {episode_reward}")

            # **🌟 Self-Play oder Gegner-Wechsel**
            if episode % 5000 == 0 and episode > 0:
                self.switch_training_strategy()

            # **💾 Modell speichern**
            if (episode + 1) % 5000 == 0:
                save_path = os.path.join(self.save_dir, f"sac_model_{episode+1}.pth")
                torch.save(self.agent.policy_net.state_dict(), save_path)
                print(f"✅ Modell gespeichert unter: {save_path}")

            # **📊 Evaluierung**
            if (episode + 1) % self.args.evaluate_every == 0:
                win_rate = self.evaluate()
                print(f"🎯 Win-Rate nach {episode+1} Episoden: {win_rate:.2f}%")

    def switch_training_strategy(self):
        """
        Schaltet zwischen verschiedenen Trainingsmethoden um:
        - Erst schwacher Gegner
        - Dann Self-Play
        - Dann stärkster gespeicherter SAC-Agent
        """
        if self.current_phase == "random":
            print("🔄 Wechsel zu Training gegen schwachen Gegner!")
            self.current_phase = "weak_opponent"
            self.opponent = BasicOpponent(weak=True)

        elif self.current_phase == "weak_opponent":
            print("🔄 Wechsel zu Self-Play!")
            self.current_phase = "self_play"
            self.opponent = self.agent  # Self-Play

        elif self.current_phase == "self_play":
            print("🔄 Wechsel zu stärkstem SAC-Agent!")
            self.current_phase = "best_sac"
            best_model = max(os.listdir(self.save_dir), key=lambda f: int(f.split("_")[-1].split(".")[0]))
            best_model_path = os.path.join(self.save_dir, best_model)
            self.agent.policy_net.load_state_dict(torch.load(best_model_path))
            self.agent.policy_net.eval()

    def evaluate(self, num_episodes=100):
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
