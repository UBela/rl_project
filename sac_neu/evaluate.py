import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super(CriticNetwork, self).__init__()
        self.layers = nn.ModuleList()
        last_size = obs_dim + action_dim

        for size in hidden_sizes:
            self.layers.append(nn.Linear(last_size, size))
            self.layers.append(nn.ReLU())
            last_size = size
        
        self.q_value = nn.Linear(last_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return self.q_value(x)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super(ActorNetwork, self).__init__()
        self.layers = nn.ModuleList()
        last_size = obs_dim

        for size in hidden_sizes:
            self.layers.append(nn.Linear(last_size, size))
            self.layers.append(nn.ReLU())
            last_size = size

        self.mu_layer = nn.Linear(last_size, action_dim)
        self.log_std_layer = nn.Linear(last_size, action_dim)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = 0.5 * (torch.tanh(log_std) + 1) * (-5 + 2) + 2  # Besser als clamping

        return mu, log_std.exp()


class SACAgent:
    def __init__(self, obs_dim, action_dim, learning_rate=3e-4, alpha=0.2, automatic_entropy_tuning=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor & Critics
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)

        # Target Critics für stabileres Training
        self.target_critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Adaptive Lernraten
        self.critic1_scheduler = optim.lr_scheduler.MultiStepLR(self.critic1_optimizer, milestones=[10000, 30000], gamma=0.5)
        self.critic2_scheduler = optim.lr_scheduler.MultiStepLR(self.critic2_optimizer, milestones=[10000, 30000], gamma=0.5)
        self.actor_scheduler = optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=[10000, 30000], gamma=0.5)

        # Entropie-Parameter
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor(action_dim, dtype=torch.float32)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = alpha

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)

        if eval_mode:
            action = mu
        else:
            action = dist.rsample()

        action = torch.tanh(action)  # Skalierung auf [-1,1]

        return action.cpu().detach().numpy()[0]

    def update_parameters(self, memory, batch_size=128, gamma=0.99, tau=0.005):
        if len(memory) < batch_size:
            return  # Falls Buffer zu klein ist, überspringe das Update

        state, action, reward, next_state, done = zip(*memory.sample(batch_size))

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            next_q1 = self.target_critic1(next_state, next_action)
            next_q2 = self.target_critic2(next_state, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * gamma * next_q

        q1_pred = self.critic1(state, action)
        q2_pred = self.critic2(state, action)
        q1_loss = nn.MSELoss()(q1_pred, target_q)
        q2_loss = nn.MSELoss()(q2_pred, target_q)

        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()

        new_action, log_prob = self.actor(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.critic1_scheduler.step()
        self.critic2_scheduler.step()
        self.actor_scheduler.step()
