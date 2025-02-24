import sys
import os
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from utils.replay_buffer import PriorityReplayBuffer
from replay_buffer import ReplayBuffer
from networks import *


class SACAgent:
    def __init__(self, state_dim, action_space, config, action_dim=4):
        self.__dict__.update(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.action_dim = action_dim
        self.hidden_dim = self.hidden_dim
        # **Replay Buffer Initialisierung**
        if self.use_PER:
            self.replay_buffer = PriorityReplayBuffer(self.buffer_size, alpha=self.per_alpha, beta=self.per_beta)
            print("Prioritized Experience Replay")
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            print("Standard Replay Buffer")

        self.policy_net = PolicyNetwork(state_dim, self.action_dim, self.action_space, self.hidden_dim, self.policy_lr, self.lr_milestones)
        
        self.qnet1 = QNetwork(state_dim, self.action_dim, self.hidden_dim, self.q_lr)
        

        # **Target Q**
        self.qnet_target = QNetwork(state_dim, self.action_dim, self.hidden_dim)

         # hard update 
        for target_param, param in zip(self.qnet_target.parameters(), self.qnet1.parameters()):
            target_param.data.copy_(param.data)
       
        #self.qnet_target.load_state_dict(self.qnet1.state_dict())

        
        # **Automatische Entropie-Anpassung**
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.tensor(self.action_dim).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.alpha_optimizer, milestones=self.alpha_milestones, gamma=0.5
            )
            

    def eval(self):
        self.qnet1.eval()
        self.qnet_target.eval()
        self.qnet_target.eval()
        self.policy_net.eval()

    def train(self):
        self.qnet1.train()
        self.qnet_target.train()
        self.qnet_target.train()
        self.policy_net.train()

    def select_action(self, state, eval_mode=False): 
        state_n = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval_mode:
            _, _, action, _ = self.policy_net.sample(state_n)
        else:
            action, _, _, _ = self.policy_net.sample(state_n)
        
        return action.detach().cpu().numpy().flatten()  


    def store_transition(self, transition: tuple):
        self.replay_buffer.add_transition(transition)

    def act(self, state):
        return self.select_action(state, True) if self.evaluate else self.select_action(state)

    def update(self, replay_buffer, batch_size, total_step):
        if self.use_PER:
            batch, tree_idxs, weights = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = replay_buffer.sample(batch_size)
            tree_idxs = None  
            weights = torch.ones(batch_size, 1).to(self.device)  
            states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device).squeeze(dim=1)
        rewards = torch.FloatTensor(np.vstack(rewards)).to(self.device).squeeze(dim=1)
        dones = torch.FloatTensor(np.vstack(dones)).to(self.device).squeeze(dim=1)

        # Berechne das Ziel-Q-Value mit Target-Q-Netzwerk
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy_net.sample(next_states)
            q1_next_targ, q2_next_targ = self.qnet_target(next_states, next_state_action)
            min_qf_next_target = torch.min(q1_next_targ, q2_next_targ) - self.alpha * next_state_log_pi
            target_q = rewards + (1 - dones) * self.gamma * min_qf_next_target.squeeze()

        # Berechne Q-Werte für das aktuelle Zustand-Aktionspaar
        q1_pred, q2_pred = self.qnet1(states, actions)
        
        td_error = torch.abs(q1_pred.squeeze() - target_q.squeeze()).detach().cpu().numpy()
        if self.use_PER:
            replay_buffer.update_priorities(tree_idxs, td_error)

        
        q1_loss = self.qnet1.compute_loss(q1_pred.squeeze(), target_q)
        q2_loss = self.qnet_target.compute_loss(q2_pred.squeeze(), target_q)
        qf_loss = q1_loss+q1_loss
        # Optimierung für Q1-Netzwerk
        self.qnet1.optimizer.zero_grad()
        qf_loss.backward() 
        self.qnet1.optimizer.step()

        pi, log_pi, _, _ = self.policy_net.sample(states)
        qf1_pi, qf2_pi = self.qnet1(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha*log_pi)-min_qf_pi).mean(axis=0)
        
        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()
        
        

        # Berechne TD-Fehler für PER (falls aktiviert)
        
        avg_priority = np.mean(td_error) if self.use_PER else 0.0
        priority_min = np.min(td_error) if self.use_PER else 0.0
        priority_max = np.max(td_error) if self.use_PER else 0.0

        # **Entropie-Update**
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha_scheduler.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)
        if total_step % self.update_target_every == 0:
            for target_param, param in zip(self.qnet_target.parameters(), self.qnet1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "td_error_mean": np.mean(td_error) if self.use_PER else 0.0,
            "td_error_std": np.std(td_error) if self.use_PER else 0.0,
            "avg_priority": avg_priority,
            "priority_min": priority_min,
            "priority_max": priority_max,
            "priority_mean": np.mean(td_error) if self.use_PER else 0.0,
            "per_beta": self.per_beta_update if self.per_beta_update else 0.0
        }


    def schedulers_step(self):
        self.policy_net.scheduler.step()
        self.qnet1.lr_scheduler.step()
