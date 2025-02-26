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
import random
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
      
        if self._config['selfplay'] and i_episode % self._config['add_self_every'] == 0:
            print("Adding agent to self-play pool...")
            new_opponent = copy.deepcopy(agent)
            new_opponent.eval()
            opponents.append(new_opponent)

    def fill_replay_buffer(self, agent, env):

        print("⏳ Filling replay buffer with random actions...")

        with open(os.devnull, 'w') as f, redirect_stdout(f):  
            with tqdm(total=self._config["buffer_size"], desc="⏳ Filling Replay Buffer", unit="samples") as pbar:
                while len(self.replay_buffer) < self._config["buffer_size"]:
                    ob, _ = env.reset()
                    obs_agent2 = env.obs_agent_two()
                    opponent = self._select_opponent([h_env.BasicOpponent(weak=True)])
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
    def train(self, agent, opponents, env, rurn_evaluation=True):
        rew_stats, q1_losses, q2_losses, actor_losses, alpha_losses = [], [], [], [], []

        lost_stats, touch_stats, won_stats = {}, {}, {}
        eval_stats = {
            'weak': {
                'reward': [],
                'touch': [],
                'won': [],
                'lost': []
            },
            'strong': {
                'reward': [],
                'touch': [],
                'won': [],
                'lost': []
            }
        }

        episode_counter = 1
        total_step_counter = 0
        grad_updates = 0
        new_op_grad = []
        while episode_counter <= self._config['max_episodes']:
            if episode_counter  == int(self._config['max_episodes']/3):
                self._config['mode'] = 'shooting'
                print("Switching to shooting mode")
            if episode_counter  == int(2*self._config['max_episodes']/3):
                self._config['mode'] = 'normal'
                print("Switching to defense mode")
            ob, _ = env.reset()
            obs_agent2 = env.obs_agent_two()
            total_reward, touched = 0, 0
            
            opponent = random.choice(opponents)
            first_time_touch = 1
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0

            opponent = random.choice(opponents)

            first_time_touch = 1
            for step in range(self._config['max_timesteps']):
                a1 = agent.act(ob)
                
                if self._config['mode'] == 'defense':
                    a2 = opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = np.zeros_like(a1)
                else:
                    a2 = opponent.act(obs_agent2)

                actions = np.hstack([a1, a2])
                next_state, reward, done, truncated, _info = env.step(actions)

                touched = max(touched, _info['reward_touch_puck'])
                step_reward = reward + 5 * _info['reward_closeness_to_puck'] - (1 - touched) * 0.1 + touched * first_time_touch * 0.1 * step
                first_time_touch = 1 - touched
                total_reward += step_reward
                agent.store_transition((ob, a1, step_reward, next_state, done))

                if touched > 0:
                    touch_stats[episode_counter] = 1

                if done:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                    break

                ob = next_state
                obs_agent2 = env.obs_agent_two()
                total_step_counter += 1

            if len(agent.replay_buffer) < self._config['batch_size']:
                continue

            for _ in range(self._config['grad_steps']):
                losses = agent.update(agent.replay_buffer, self._config['batch_size'], self.total_steps)
                grad_updates += 1
                q1_losses.append(losses['q1_loss'])
                q2_losses.append(losses['q2_loss'])
                actor_losses.append(losses['policy_loss'])
                alpha_losses.append(losses['alpha_loss'])

            if self._config['selfplay']:
                    if (
                        grad_updates % self._config['add_self_every'] == 0
                    ):
                        new_opponent = copy.deepcopy(agent)
                        new_opponent.eval()
                        opponents.append(new_opponent)
                        new_op_grad.append(grad_updates)
            agent.schedulers_step()
            self.logger.info(f"Episode {episode_counter} finished after {step} steps with total reward {total_reward} winner {env.winner}")
            

            if episode_counter % self._config['evaluate_every'] == 0:
                with open(os.devnull, 'w') as f, redirect_stdout(f): 
                    agent.eval()
                
                for eval_op in ['strong', 'weak']:
                    ev_opponent = opponents[0] if eval_op == 'strong' else h_env.BasicOpponent(False)
                    with open(os.devnull, 'w') as f, redirect_stdout(f):     
                        rew, touch, won, lost = evaluate(agent, env, ev_opponent, 100)
                    eval_stats[eval_op]['reward'].append(rew)
                    eval_stats[eval_op]['touch'].append(touch)
                    eval_stats[eval_op]['won'].append(won)
                    eval_stats[eval_op]['lost'].append(lost)
                agent.train()
            
            if episode_counter % self._config["log_interval"] == 0:
                trainingmetrics = {
                    'q1_loss': np.mean(q1_losses),
                    'q2_loss': np.mean(q2_losses),
                    'policy_loss': np.mean(actor_losses),
                    'alpha_loss': np.mean(alpha_losses),
                    'total_reward': total_reward,
                    'touch': touch_stats[episode_counter],
                    'won': won_stats[episode_counter],
                    'lost': lost_stats[episode_counter],
                    'total_steps': total_step_counter,
                    'grad_updates': grad_updates}
                self.logger.log_training(episode_counter, trainingmetrics)
                self.logger.save_model(agent, episode_counter)
                #self.logger.info(f"Episode {episode_counter} evaluation: {eval_stats}")
                

            rew_stats.append(total_reward)

            
            episode_counter += 1
        self.logger.info('Training finished.')
        
        self.logger.info(f'Reward: {rew_stats} Touch: {touch_stats} Won: {won_stats} Lost: {lost_stats}')
        
        self.logger.info('Saving training statistics...')

        # Plot reward
        #self.logger.plot_running_mean(data=rew_stats, title='Total reward', filename='total-reward.pdf', show=False)

        # Plot evaluation stats
        #self.logger.plot_evaluation_stats(eval_stats, self._config['evaluate_every'], 'evaluation-won-lost.pdf')

        # Plot losses
        #for loss, title in zip([q1_losses, q2_losses, actor_losses, alpha_losses],
                               #['Q1 loss', 'Q2 loss', 'Policy loss', 'Alpha loss']):
            #self.logger.plot_running_mean(
                #data=loss,
                #title=title,
                #filename=f'{title.replace(" ", "-")}.pdf',
                #show=False,
                #v_milestones=new_op_grad,
            #)

        # Save agent
        self.logger.save_model(agent, 'agent')

        #if self.run_evaluation:
            #agent.eval()
            #agent._config['show'] = True
        evaluate(agent, env, h_env.BasicOpponent(weak=False), 200)
