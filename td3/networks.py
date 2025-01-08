import torch
import numpy as np

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, act_fn=torch.nn.Tanh(), out_act_fn=None):   
        super(FeedForwardNetwork, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.output_activation = out_act_fn
        
        layer_sizes = [self.input_dim] + self.hidden_dims
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.act_fns = [act_fn for _ in self.layers]
        self.readout = torch.nn.Linear(self.hidden_dims[-1], self.output_dim)
        
    def forward(self, x):
        for layer, act_fn in zip(self.layers, self.act_fns):
            x = act_fn(layer(x))
        x = self.readout(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
        

class QFunction(FeedForwardNetwork):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims=[100,100], lr = 0.0002):
        super(QFunction, self).__init__(input_dim=observation_dim + action_dim, hidden_dims=hidden_dims, output_dim=1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-6)
        self.loss = torch.nn.SmoothL1Loss()
        
    def fit(self, observations: torch.tensor, actions: torch.tensor, targets: torch.tensor):
        self.train()
        self.optimizer.zero_grad()
        
        pred = self.forward(torch.cat([observations, actions], dim=1))
        
        loss = self.loss(pred, targets)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def Q_value(self, observations: torch.tensor, actions: torch.tensor):
        return self.forward(torch.cat([observations, actions], dim=1))
        