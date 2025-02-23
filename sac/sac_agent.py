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


# **Wert-Netzwerk (Value Network)**
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.layers(state)


# **Q-Wert-Netzwerk**
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.layers(x)


# **Policy-Netzwerk (Gaussian Policy)**
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.layers(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mean, log_std

    def sample(self, state, eval_mode=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample() if not eval_mode else mean
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


# **SAC-Agent**
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, tau, alpha,
                 automatic_entropy_tuning, policy_lr, q_lr, value_lr, alpha_lr,
                 buffer_size, per_alpha, per_beta, per_beta_update, alpha_milestones,
                 use_PER, device, results_folder):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.use_PER = use_PER
        self.results_folder = results_folder
        self.alpha_lr = alpha_lr
        self.alpha_milestones = alpha_milestones
        self.per_beta_update = per_beta_update

        os.makedirs(self.results_folder, exist_ok=True)

        # **Replay Buffer Initialisierung**
        if use_PER:
            self.replay_buffer = PriorityReplayBuffer(buffer_size, alpha=per_alpha, beta=per_beta)
            print("Prioritized Experience Replay")
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
            print("Standard Replay Buffer")

        # **Netzwerke initialisieren**
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # **Optimierer**
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.q_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(self.q_optimizer1, milestones=[10000, 30000], gamma=0.5)
        self.q_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.q_optimizer2, milestones=[10000, 30000], gamma=0.5)
        self.policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[10000, 30000], gamma=0.5)


        # **Soft Target Update**
        self.soft_tau = tau

        # **Entropie-Anpassung**
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

            if isinstance(alpha_milestones, str):
                alpha_milestones = [int(x) for x in alpha_milestones.split(' ')]

            self.alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.alpha_optimizer, milestones=alpha_milestones, gamma=0.5
            )
        # **Target-Netzwerk Gewichte synchronisieren**
        self.update_target_network(soft_update=False)

    def update_target_network(self, soft_update=True):
        """ Kopiert die Gewichte ins Target-Netzwerk mit Soft-Update. """
        tau = self.soft_tau if soft_update else 1.0
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state, eval_mode)
        return action.cpu().numpy().flatten()
    def eval(self):
 
        self.value_net.eval()
        self.target_value_net.eval()
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()
    def train(self):

        self.value_net.train()
        self.target_value_net.train()
        self.q_net1.train()
        self.q_net2.train()
        self.policy_net.train()


    def act(self, state):
        return self.select_action(state)
    def schedulers_step(self):
        """Aktualisiert die Lernraten-Scheduler für Q-Netzwerke, Policy und Alpha (falls aktiv)."""
        if hasattr(self, "q_scheduler1"):
            self.q_scheduler1.step()
        if hasattr(self, "q_scheduler2"):
            self.q_scheduler2.step()
        if hasattr(self, "policy_scheduler"):
            self.policy_scheduler.step()
        if self.automatic_entropy_tuning and hasattr(self, "alpha_scheduler"):
            self.alpha_scheduler.step()


    def update(self, replay_buffer, batch_size):
        if self.use_PER:
            batch, tree_idxs, weights = replay_buffer.sample(batch_size)
            #print(f"[DEBUG] Sampled Indices: {tree_idxs[:10]}")  # Zeigt die ersten 10 ausgewählten Indizes
            #print(f"[DEBUG] Sampled Weights: {weights[:10]}")
            states, actions, rewards, next_states, dones = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
        else:
            batch = replay_buffer.sample(batch_size)
            tree_idxs = None  
            weights = torch.ones(batch_size, 1).to(self.device)  

            states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        rewards = torch.FloatTensor(np.vstack(rewards)).to(self.device)
        dones = torch.FloatTensor(np.vstack(dones)).to(self.device)

        rewards = torch.clamp(rewards, -10, 10)

        # **Soft-Q-Zielwert**
        with torch.no_grad():
            next_v = self.target_value_net(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        # **Q-Wert Update**
        q1_loss = nn.MSELoss()(self.q_net1(states, actions), target_q)
        q2_loss = nn.MSELoss()(self.q_net2(states, actions), target_q)

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        # **Policy-Netzwerk Update**
        new_actions, log_probs = self.policy_net.sample(states)
        min_q = torch.min(self.q_net1(states, new_actions), self.q_net2(states, new_actions))
        policy_loss = (self.alpha * log_probs - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.use_PER:
            with torch.no_grad():
                td_errors = torch.abs(self.q_net1(states, actions) - target_q).detach().cpu().numpy()
            #print(f"[DEBUG] Updating priorities for indices: {tree_idxs[:10]}")
            replay_buffer.update_priorities(tree_idxs, td_errors)

            avg_priority = np.mean(td_errors)
            priority_min = np.min(td_errors)
            priority_max = np.max(td_errors)
        else:
            avg_priority, priority_min, priority_max = None, None, None

        # **Entropie-Update**
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            self.alpha_scheduler.step()  # Lernrate für Alpha-Optimierer anpassen
        else:
            alpha_loss = torch.tensor(0.0)

        # **Target-Netzwerk Update**
        self.update_target_network()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "td_error_mean": np.mean(td_errors) if self.use_PER else 0.0,
            "td_error_std": np.std(td_errors) if self.use_PER else 0.0,
            "avg_priority": avg_priority if self.use_PER else 0.0,
            "priority_min": priority_min if self.use_PER else 0.0,
            "priority_max": priority_max if self.use_PER else 0.0,
            "priority_mean": np.mean(td_errors) if self.use_PER else 0.0,
            "per_beta": self.per_beta_update if self.per_beta_update else 0.0
        }
