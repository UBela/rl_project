import sys
import os
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *
from torch.distributions import Normal
from utils.replay_buffer import PriorityReplayBuffer
from replay_buffer import ReplayBuffer
import json
os.makedirs("logs", exist_ok=True)
log_file = open("logs/training_log.txt", "a")

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)  # Neue Schicht
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return self.linear4(x)


# **Q-Wert-Netzwerk**
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.norm1(self.linear1(x)))
        x = torch.relu(self.norm2(self.linear2(x)))
        x = torch.relu(self.linear3(x))
        return self.linear4(x)

# 🎯 **Policy-Netzwerk**
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

# 🚀 **SAC-Agent**
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, tau, alpha,
                 automatic_entropy_tuning, policy_lr, q_lr, value_lr, 
                 buffer_size, per_alpha, per_beta, per_beta_update, 
                 use_PER, device, results_folder):

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.use_PER = use_PER
        self.results_folder = results_folder 

        os.makedirs(self.results_folder, exist_ok=True)
        self.training_log_filename = os.path.join(self.results_folder, "training_log.json")

        # **Falls die Datei nicht existiert, erstelle eine leere JSON-Datei**
        if not os.path.exists(self.training_log_filename):
            with open(self.training_log_filename, "w") as f:
                json.dump([], f)
        # 🏆 **Replay Buffer wählen (Prioritized oder normal)**
        if use_PER:
            self.replay_buffer = PriorityReplayBuffer(buffer_size, alpha=per_alpha, beta=per_beta)
            print("PER True")
        else:
            print("PER False")
            self.replay_buffer = ReplayBuffer(buffer_size)

        # 🏗️ **Netzwerke erstellen**
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # ⚙️ **Optimierer**
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.policy_update_freq = 2
        '''self.scheduler_q = optim.lr_scheduler.StepLR(self.soft_q_optimizer1, step_size=5000, gamma=0.5)
        self.scheduler_q2 = optim.lr_scheduler.StepLR(self.soft_q_optimizer2, step_size=5000, gamma=0.5)
        self.scheduler_policy = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=5000, gamma=0.5)'''

        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        
        if self.automatic_entropy_tuning:
            self.target_entropy = -0.7*torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)
            print(f"log_alpha: {self.log_alpha.item()}, alpha: {self.alpha}")

        #  **PER Beta-Update**
        self.per_beta_update = per_beta_update
    def _log_training_data(self, log_data):
        """Speichert Trainingsdaten als JSON in der Log-Datei und fängt Fehler ab."""
        try:
            with open(self.training_log_filename, "r+") as f:
                try:
                    logs = json.load(f)  # Existierende Daten laden
                except json.JSONDecodeError:  
                    logs = []  # Falls Fehler, starte mit leerem JSON

                logs.append(log_data)
                f.seek(0)  # Zurück an den Anfang der Datei
                json.dump(logs, f, indent=4)
                f.truncate()  # Falls die Datei vorher größer war, kürzen
        except Exception as e:
            #print(f"⚠️ Fehler beim Schreiben der Log-Datei: {e}")

            """ Speichert Trainingsdaten als JSON in `training_log.json` """
            with open(self.training_log_filename, "r+") as f:
                logs = json.load(f)
                logs.append(log_data)
                f.seek(0)
                json.dump(logs, f, indent=4)
    #  **Aktion wählen**
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)
        noise = np.random.normal(0, 0.2, size=4)
        action = action.cpu().numpy().flatten()[:4] + noise
        #print(action.shape)
        return np.clip(action, -1, 1)
    def act(self, state):
        return self.select_action(state)
    
    # **SAC Update**
    def update(self, replay_buffer, batch_size):
        if self.use_PER:
            batch, tree_idxs, weights = replay_buffer.sample(batch_size)
            state, action, reward, next_state, done = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
        else:
            batch = replay_buffer.sample(batch_size)
            tree_idxs = None  
            weights = torch.ones(batch_size, 1).to(self.device)  

            state, action, reward, next_state, done = zip(*batch)

        # 🔹 Tensor Umwandlung
        state = torch.FloatTensor(np.vstack(state)).to(self.device)
        next_state = torch.FloatTensor(np.vstack(next_state)).to(self.device)
        action = torch.FloatTensor(np.vstack(action)).to(self.device)
        reward = torch.FloatTensor(np.vstack(reward)).to(self.device)
        done = torch.FloatTensor(np.vstack(done)).to(self.device)

        # 🔥 **Verbesserung: Reward Clipping & Normalisierung**
        reward = torch.clamp(reward, -10, 10)  
        reward = (reward - reward.mean()) / (reward.std() + 1e-6)  # Stabilität verbessern

        # 🎯 **Zielwert-Berechnung**
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * (target_value - 0.002 * torch.randn_like(target_value).detach())

        # 🔹 **Q-Werte Berechnung**
        q1_pred = self.soft_q_net1(state, action)
        q2_pred = self.soft_q_net2(state, action)

        # 📌 **PER Gewichtung**
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)  
        q1_loss = (weights * nn.functional.mse_loss(q1_pred, target_q_value.detach(), reduction='none')).mean()
        q2_loss = (weights * nn.functional.mse_loss(q2_pred, target_q_value.detach(), reduction='none')).mean()

        # 🔥 **Verbesserung: Q-Loss Clamping für Stabilität**
        q1_loss = torch.clamp(q1_loss, -1, 1)
        q2_loss = torch.clamp(q2_loss, -1, 1)

        # 🚀 **Optimierung Q-Netzwerke**
        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        # 🎯 **Policy-Netzwerk Update**
        new_action, log_prob = self.policy_net.sample(state)
        min_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (log_prob - min_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 🏆 **Verbesserung: Adaptive Exploration-Noise für Action Auswahl**
        noise = np.random.normal(0, 0.2, size=4)  # **Nur für die ersten 4 Actions!**
        action = action.cpu().numpy().flatten()[:4] + noise  # **Gegner wird nicht beeinflusst!**

        # 📊 **Logging Daten sammeln**
        log_data = {
            "td_error_mean": q1_loss.item(),
            "td_error_std": q2_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "log_prob_mean": log_prob.mean().item(),
            "log_prob_std": log_prob.std().item(),
            "alpha": self.alpha if self.automatic_entropy_tuning else None
        }
        self._log_training_data(log_data)

        # 🔄 **Prioritäten für PER updaten**
        if self.use_PER:
            td_errors = torch.abs(q1_pred - target_q_value).detach().cpu().numpy()
            td_errors = torch.clamp(torch.tensor(td_errors, device=self.device, dtype=torch.float32), -1, 1)
            replay_buffer.update_priorities(tree_idxs, td_errors)

        # 🔥 **Alpha Entropy-Tuning Update**
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        return [q1_loss.item(), q2_loss.item(), policy_loss.item()]
