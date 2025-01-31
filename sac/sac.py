import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import copy

# ----------------- 🎯 Replay Buffer -----------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------- 🔥 Netzwerke -----------------
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


# ----------------- 🚀 SAC Agent -----------------
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        policy_lr=3e-4,
        q_lr=3e-4,
        value_lr=3e-4,
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
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

        # Kopiere die Netzwerkgewichte auf das Target-Netzwerk
        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)  # Policy wählt eine Aktion
        return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Berechnung des Q-Wertes
        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value
        q1_loss = F.mse_loss(self.soft_q_net1(state, action), next_q_value.detach())
        q2_loss = F.mse_loss(self.soft_q_net2(state, action), next_q_value.detach())

        # Update der Q-Netzwerke
        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        # Berechnung der Policy-Verlustfunktion
        new_action, log_prob = self.policy_net.sample(state)
        q_value = torch.min(
            self.soft_q_net1(state, new_action),
            self.soft_q_net2(state, new_action),
        )
        policy_loss = (log_prob * self.alpha - q_value).mean()

        # Update des Policy-Netzwerks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Berechnung des Value-Netzwerk-Verlustes
        expected_value = self.value_net(state)
        value_loss = F.mse_loss(
            expected_value,
            (q_value - log_prob * self.alpha).detach(),
        )

        # Update des Value-Netzwerks
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update des Target-Value-Netzwerks
        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1 - self.tau) + param.data * self.tau
            )

