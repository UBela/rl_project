import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Actor_Net(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, lr=0.0002):   
        super(Actor_Net, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        layer_sizes = [self.input_dim] + self.hidden_dims
        self.layers = nn.ModuleList([torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.act_fns = [nn.ReLU() for _ in self.layers]
        self.readout =  nn.Linear(self.hidden_dims[-1], self.output_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    def forward(self, x):
        for layer, act_fn in zip(self.layers, self.act_fns):
            x = act_fn(layer(x))
        x = self.readout(x)
        
        return F.tanh(x)
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
        

class Critic_Net(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims=[100, 100], lr=0.0002):
        super(Critic_Net, self).__init__()
        self.input_dim = observation_dim + action_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 1  
        layer_sizes = [self.input_dim] + self.hidden_dims
        self.layers_1 = nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layers_2 = nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.act_fns = [nn.ReLU() for _ in self.layers_1]
        self.readout_1 = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.readout_2 = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x):
        x1 = x
        x2 = x
        for layer, act_fn in zip(self.layers_1, self.act_fns):
            x1 = act_fn(layer(x1))
        
        for layer, act_fn in zip(self.layers_2, self.act_fns):
            x2 = act_fn(layer(x2))
        x1 = self.readout_1(x1)
        x2 = self.readout_2(x2)
        return x1, x2

    def Q_value(self, observations: torch.tensor, actions: torch.tensor):
    
        inputs = torch.cat([observations, actions], dim=1)
        q1, _ = self.forward(inputs)
        return q1
    
    def predict(self, observations: np.ndarray, actions: np.ndarray):
        with torch.no_grad():
            return self.forward(torch.from_numpy(np.concatenate([observations, actions], axis=1).astype(np.float32))).numpy()

        