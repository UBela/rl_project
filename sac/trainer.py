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
from td3.utils import *
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

    def _select_opponent(self, opponents):
        return np.random.choice(opponents)  

    def _add_self_play_agent(self, agent, opponents, i_episode):
      
        if not self._config['use_self_play']:
            return
        if i_episode >= self._config['self_play_start'] and self.total_gradient_steps % self._config['self_play_intervall'] == 0:
            print("Adding agent to self-play opponents...")
            opponents.append(copy.deepcopy(agent))

    def fill_replay_buffer(self, agent, env):  
        
        while len(agent.replay_buffer) < self._config['buffer_size']:
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            opponent = self._select_opponent(opponents=[h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)])
            done = False
            trunc = False
            while not (done or trunc):
                a1 = np.random.uniform(-1, 1, 4)
                a2 = opponent.act(obs_agent2)
                actions = np.hstack([a1, a2])
                (ob_new, reward, done, trunc, _info) = env.step(actions)
                
                agent.store_transition((ob, a1, reward, ob_new, done))
                ob = ob_new
                obs_agent2 = env.obs_agent_two()

        print(f"Replay buffer filled with {len(self.replay_buffer)} samples.")

    def train(self, agent, opponents, env):
 
        rew_stats, q1_losses, q2_losses, actor_losses, alpha_losses = [], [], [], [], []
        lost_stats, touch_stats, won_stats = {}, {}, {}
        eval_stats = {"weak": {"reward": [], "touch": [], "won": [], "lost": []},
                      "strong": {"reward": [], "touch": [], "won": [], "lost": []}}
        episode_counter, total_step_counter, grad_updates = 1, 0, 0
        random_seed = self._config['random_seed']
        iter_fit = self._config['iter_fit']
        log_interval = self._config['log_interval']
        max_episodes = self._config['max_episodes']
        max_timesteps = self._config['max_timesteps']
        render = self._config['render']
        random_seed = self._config['random_seed']
        use_hard_opp = self._config['use_hard_opp']
        evaluate_every = self._config['evaluate_every']
        rewards = []
        lengths = []
        wins_per_episode = {}
        loses_per_episode = {}
        losses = []
        wins_per_episode_eval_easy = {}
        loses_per_episode_eval_easy = {}
        wins_per_episode_eval_hard = {}
        loses_per_episode_eval_hard = {}
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            env.seed(random_seed)
        
        if self._config['use_PER']:
            self.fill_replay_buffer(agent, env)
            print("Replay buffer filled with initial samples.")

        for i_episode in range(1, max_episodes + 1):
                # 50% chance that the agent is player 1
            agent_is_player_1 = np.random.choice([True, False])
            agent_is_player_1 = True
            
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            
            total_reward = 0
            wins_per_episode[i_episode] = 0
            loses_per_episode[i_episode] = 0
            
            if self._config['use_self_play']:
                opponent = self._select_opponent(opponents)
            else:
                opponent = opponents[1]
            
            if self._config['use_PER']:
                if self._config['per_beta_update'] is None:
                    beta_update = (1.0 - self._config["per_beta"]) / max_episodes
                else: 
                    beta_update = self._config['per_beta_update']
                agent.update_per_beta(beta_update)
            
            for t in range(max_timesteps):
                
                if agent_is_player_1:
                    a1 = agent.act(ob)
                    a2 = opponent.act(obs_agent2)
                    actions = np.hstack([a1, a2])
                else:
                    a1 = opponent.act(obs_agent2)
                    a2 = agent.act(ob)
                    actions = np.hstack([a1, a2])
                
                (ob_new, reward, done, trunc, _info) = env.step(actions)
                #if render: env.render()
                
                reward = reward_player_2(env) if not agent_is_player_1 else reward

                if agent_is_player_1:

                    agent.store_transition((ob, a1, reward, ob_new, done))
                else:
                    agent.store_transition((ob, a2, reward, ob_new, done))
                
                total_reward += reward
                ob_new_copy = ob_new  
                if agent_is_player_1:
                    ob = ob_new
                    obs_agent2 = env.obs_agent_two()
                else:
                    ob = ob_new_copy  
                    obs_agent2 = env.obs_agent_two()

                self.total_steps += 1
                

                if done or trunc:
                    winner = _info.get('winner', None)
                    if agent_is_player_1:
                        wins_per_episode[i_episode] = 1 if winner == 1 else 0
                        loses_per_episode[i_episode] = 1 if winner == -1 else 0
                    else:
                        wins_per_episode[i_episode] = 1 if winner == -1 else 0
                        loses_per_episode[i_episode] = 1 if winner == 1 else 0
                    break
            losses.extend(agent.train(iter_fit=iter_fit))
            rewards.append(total_reward)
            lengths.append(t)
            training_metrics = {
                "td_error_mean": 0.0,
                "td_error_std": 0.0,
                "q1_loss": 0.0,
                "q2_loss": 0.0,
                "policy_loss": 0.0,
                "log_prob_mean": 0.0,
                "log_prob_std": 0.0,
                "alpha": agent.alpha if agent.automatic_entropy_tuning else 0.0,
                "alpha_loss": 0.0,
                "avg_priority": 0.0,
                "priority_min": 0.0,
                "priority_max": 0.0,
                "priority_mean": 0.0,
                "per_beta": 0.0

            }
            # Konvertiere alle Werte zu float, ersetze None mit 0.0
            training_metrics = {k: float(v) if v is not None else 0.0 for k, v in training_metrics.items()}

            q1_loss, q2_loss, policy_loss, alpha_loss = 0.0, 0.0, 0.0, 0.0  
            if len(agent.replay_buffer) >= self._config['batch_size']:
                for _ in range(self._config['grad_steps']):
                    losses = agent.update(agent.replay_buffer, self._config['batch_size'])

                    q1_loss, q2_loss, policy_loss, alpha_loss = (
                        losses["q1_loss"],
                        losses["q2_loss"],
                        losses["policy_loss"],
                        losses["alpha_loss"],
                    )
                if agent.use_PER:
                    training_metrics["td_error_mean"] = losses.get("td_error_mean", 0.0)
                    training_metrics["td_error_std"] = losses.get("td_error_std", 0.0)
                    training_metrics["avg_priority"] = losses.get("avg_priority", 0.0)
                    training_metrics["priority_min"] = losses.get("priority_min", 0.0)
                    training_metrics["priority_max"] = losses.get("priority_max", 0.0)
                    training_metrics["priority_mean"] = losses.get("priority_mean", 0.0)
                    training_metrics["per_beta"] = losses.get("per_beta", 0.0)
            grad_updates += 1
            q1_losses.append(q1_loss)
            q2_losses.append(q2_loss)
            actor_losses.append(policy_loss)
            alpha_losses.append(alpha_loss)

            # Update `training_metrics` mit den neuen Werten
            training_metrics["q1_loss"] = q1_losses[-1] if len(q1_losses) > 0 else 0
            training_metrics["q2_loss"] = q2_losses[-1] if len(q2_losses) > 0 else 0
            training_metrics["policy_loss"] = actor_losses[-1] if len(actor_losses) > 0 else 0
            training_metrics["log_prob_mean"] = np.mean(alpha_losses[-100:]) if len(alpha_losses) > 0 else 0
            training_metrics["log_prob_std"] = np.std(alpha_losses[-100:]) if len(alpha_losses) > 0 else 0
            training_metrics["alpha"] = agent.alpha if agent.automatic_entropy_tuning else 0.0

            
            if self._config['lr_milestones']:
                agent.schedulers_step()
            self.total_gradient_steps += iter_fit
            self._add_self_play_agent(agent, opponents, i_episode)    
            # Konvertiere `training_metrics` Werte in native Python-Datentypen
            training_metrics = {k: float(v) for k, v in training_metrics.items()}

            # Dann loggen
            self.logger.print_episode_info(episode_counter, step, total_reward, env.winner, training_metrics)

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
        

        print("✅ Training abgeschlossen!")
