import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *
from torch.distributions import Normal
from utils.replay_buffer import ReplayBuffer, PriorityReplayBuffer

# 🔥 **Wert-Netzwerk**
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

# 🔥 **Q-Wert-Netzwerk**
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

# 🎯 **Policy-Netzwerk**
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
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
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99, tau=0.005, alpha=0.0,
                 automatic_entropy_tuning=False, policy_lr=1e-4, q_lr=1e-3, value_lr=1e-3, 
                 buffer_size=int(2**20), per_alpha=0.6, per_beta=0.4, per_beta_update=None, use_PER=True, device="cpu"):
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.use_PER = use_PER

        # 🏆 **Replay Buffer wählen (Prioritized oder normal)**
        if use_PER:
            self.replay_buffer = PriorityReplayBuffer(buffer_size, alpha=per_alpha, beta=per_beta)
        else:
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

        # 📌 **Target Value Netzwerke kopieren**
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        # 🔥 **Automatische Alpha-Anpassung**
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)

        # 📈 **PER Beta-Update**
        self.per_beta_update = per_beta_update

    # 🎯 **Aktion wählen**
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)
        return action.cpu().numpy().flatten()

    # 🔄 **SAC Update**
    def update(self, replay_buffer, batch_size):
        if self.use_PER:
            batch, tree_idxs, weights = replay_buffer.sample(batch_size)
        else:
            batch = replay_buffer.sample(batch_size)
            weights = torch.ones(batch_size, 1).to(self.device)  

        state, action, reward, next_state, done = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
        
        state = torch.FloatTensor(np.vstack(state)).to(self.device)
        next_state = torch.FloatTensor(np.vstack(next_state)).to(self.device)
        action = torch.FloatTensor(np.vstack(action)).to(self.device)
        reward = torch.FloatTensor(np.vstack(reward)).to(self.device)
        done = torch.FloatTensor(np.vstack(done)).to(self.device)

        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value

        q1_pred = self.soft_q_net1(state, action)
        q2_pred = self.soft_q_net2(state, action)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)  
        q1_loss = (weights * nn.functional.mse_loss(q1_pred, target_q_value.detach(), reduction='none')).mean()
        q2_loss = (weights * nn.functional.mse_loss(q2_pred, target_q_value.detach(), reduction='none')).mean()

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        new_action, log_prob = self.policy_net.sample(state)
        min_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (log_prob - min_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.use_PER:
            td_errors = torch.abs(q1_pred - target_q_value).detach().cpu().numpy()
            replay_buffer.update_priorities(tree_idxs, td_errors)

        return [q1_loss.item(), q2_loss.item(), policy_loss.item()]
