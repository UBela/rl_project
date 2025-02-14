import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../td3')))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from replay_buffer import ReplayBuffer

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
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
        return action, log_prob

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, tau, policy_lr, q_lr, value_lr, device):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)
        return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value
        q1_loss = F.mse_loss(self.soft_q_net1(state, action), target_q_value.detach())
        q2_loss = F.mse_loss(self.soft_q_net2(state, action), target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        new_action, log_prob = self.policy_net.sample(state)
        policy_loss = (log_prob - torch.min(self.soft_q_net1(state, new_action),
                                            self.soft_q_net2(state, new_action))).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
