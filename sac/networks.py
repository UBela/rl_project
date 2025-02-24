import torch.nn as nn
import torch
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
    
class QNetwork(nn.Module):
      
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256], lr=3e-4, weight_decay=0.0, lr_factor=0.5):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        q1_layers = [nn.Linear(state_dim + action_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            q1_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            q1_layers.append(nn.ReLU())
        q1_layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.q1_net = nn.Sequential(*q1_layers)
        q2_layers = [nn.Linear(state_dim + action_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            q2_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            q2_layers.append(nn.ReLU())
        q2_layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.q2_net = nn.Sequential(*q2_layers)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=lr_factor)
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1_value = self.q1_net(x)
        q2_value = self.q2_net(x)
        return q1_value, q2_value
    
    def compute_loss(self, q_values, target_q_values):
        return self.loss_fn(q_values, target_q_values)
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_space, hidden_dim, learning_rate=3e-4, lr_milestones=None, lr_factor=0.5, reparam_noise=1e-6):
        super(PolicyNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.reparam_noise = reparam_noise
        self.action_space = action_space
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=0.000001)
        if lr_milestones:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=lr_milestones, gamma=lr_factor
            )
        else:
            self.scheduler = None
        if self.action_space is not None:
            low, high = self.action_space.low[action_dim], self.action_space.high[:action_dim]
            self.action_scale = torch.tensor((high - low) / 2.,  dtype=torch.float32, device=self.device)
            self.action_bias = torch.tensor((high + low) / 2., dtype=torch.float32, device=self.device)
        else:
            self.action_scale = torch.ones(action_dim, dtype=torch.float32, device=self.device)
            self.action_bias = torch.zeros(action_dim, dtype=torch.float32, device=self.device)
        self.to(self.device)

    def forward(self, state):
        x = self.layers(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 10)
        return mean, log_std

    def sample(self, state, eval_mode=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)


        z = normal.rsample() if not eval_mode else mean
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, log_std
    
    def step_scheduler(self):
        if self.scheduler:
            self.scheduler.step()