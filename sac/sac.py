import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

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
            np.array(reward).reshape(-1, 1),
            np.array(next_state),
            np.array(done).reshape(-1, 1),
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
        auto_alpha=True,
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.action_dim = action_dim

        # Netzwerke
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Temperaturparameter (Alpha)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)
        else:
            self.alpha = alpha

        # Ziel-Netzwerk initialisieren
        self.update_target_network(soft_update=False)

    def update_target_network(self, soft_update=True):
        """ Aktualisiert das Target Value-Netzwerk mit Soft Updates oder komplettem Kopieren """
        tau = self.tau if soft_update else 1.0
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

    def select_action(self, state, eval=False):
        """ Wählt eine Aktion basierend auf der aktuellen Policy """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)
        return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        """ Aktualisiert das SAC-Modell basierend auf einer Minibatch """
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Q-Werte berechnen
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value
        target_q_value = target_q_value.detach()

        q1_loss = F.mse_loss(self.soft_q_net1(state, action), target_q_value)
        q2_loss = F.mse_loss(self.soft_q_net2(state, action), target_q_value)

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        # Policy Loss berechnen
        new_action, log_prob = self.policy_net.sample(state)
        q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (log_prob * self.alpha - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Berechnung des Value-Netzwerk-Verlustes
        expected_value = self.value_net(state)
        value_loss = F.mse_loss(
            expected_value,
            (q_value - log_prob * self.alpha).detach(),
        )

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Soft Updates für das Target-Value-Netzwerk
        self.update_target_network()

        # Automatische Anpassung des Entropie-Koeffizienten (Alpha)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
