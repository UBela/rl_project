import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gymnasium as gym
import numpy as np
from noise import OUNoise, GaussianNoise
from networks import QFunction, FeedForwardNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class TD3Agent(object):
    def __init__(self, observation_space, action_space, **userconfig):
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        
        self.config = {
            "eps": 0.1,
            "actor_lr": 0.00001,
            "critic_lr": 0.0001,
            "discount": 0.95,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "batch_size": 100,
            "hidden_dim_critic": [128, 128,64],
            "hidden_dim_actor": [128, 128],
            "noise": "Gaussian",
            "noise_clip": 0.5,
            "noise_std":0.2,
            "policy_update_freq": 2,
            "update_target_freq": 100,
            "buffer_size": int(1e6),
            "max_sigma": 1.0,
            "min_sigma": 1.0,
            "decay_period": int(1e6)
        }
        
        self._config.update(userconfig)    
        self._eps = self._config["eps"]
        self._noise_clamp = self._config["noise_clip"]
        self._max_sigma = self._config["max_sigma"]
        self._min_sigma = self._config["min_sigma"]
        self._decay_period = self._config["decay_period"]   
        self.replay_buffer = ReplayBuffer(max_size=self._config["buffer_size"])
        
        self.Q_net_1 = QFunction(observation_dim=self._observation_space.shape[0], 
                               action_dim=self._action_n, 
                               hidden_dims=self._config["hidden_dim_critic"],
                               learning_rate=self._config["critic_lr"])
        self.Q_net_2 = QFunction(observation_dim=self._observation_space.shape[0],
                                 action_dim=self._action_n, 
                                 hidden_dims=self._config["hidden_dim_critic"],
                                 learning_rate=self._config["critic_lr"])
        self.Q_target_1 = QFunction(observation_dim=self._observation_space.shape[0],
                                    action_dim=self._action_n, 
                                    hidden_dims=self._config["hidden_dim_critic"],
                                    learning_rate= 0)
        
        self.Q_target_2 = QFunction(observation_dim=self._observation_space.shape[0],
                                    action_dim=self._action_n,
                                    hidden_dims=self._config["hidden_dim_critic"],
                                    learning_rate= 0)
        self.policy_net = FeedForwardNetwork(input_dim=self._observation_space.shape[0],
                                               hidden_dims=self._config["hidden_dim_actor"],
                                               output_dim=self._action_n,
                                               out_act_fn=torch.nn.Tanh(),
                                               act_fn=torch.nn.ReLU(),
                                               )
        
        self.policy_target = FeedForwardNetwork(input_dim=self._observation_space.shape[0],
                                                    hidden_dims=self._config["hidden_dim_actor"],
                                                    output_dim=self._action_n,
                                                    out_act_fn=torch.nn.Tanh(),
                                                    act_fn=torch.nn.ReLU(),
                                                    )
        
        self._copy_nets()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self._config["actor_lr"], eps=1e-6)
        
        self.train_iter = 0 
        
    def _copy_nets(self):
        self.Q_target_1.load_state_dict(self.Q_net_1.state_dict())
        self.Q_target_2.load_state_dict(self.Q_net_2.state_dict())
        self.policy_target.load_state_dict(self.policy_net.state_dict())
        
    def get_action(self, observation, t=0):
        action = self.policy_net.predict(observation)
        sigma = self._max_sigma - ((self._max_sigma - self._min_sigma) * min(t / self._decay_period, 1.0))
        action = action + np.random.normal(size=action.shape) * sigma
        return np.clip(action, self._action_space.low, self._action_space.high)
            
    def store_transition(self, transition):
        self.replay_buffer.add_transition(transition)
        
    def state(self):
        return (self.Q_net_1.state_dict(),
                self.Q_net_2.state_dict(),
                self.policy_net.state_dict())
    
    def restore_state(self, state):
        self.Q_net_1.load_state_dict(state[0])
        self.Q_net_2.load_state_dict(state[1])
        self.policy_net.load_state_dict(state[2])
        self._copy_nets()
        
    def _sliding_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self._config["tau"] * param.data + (1.0 - self._config["tau"]) * target_param.data)
            
    def train(self, iter_fit: int = 32):
        
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = [] 
        self.train_iter += 1
        
        for _ in range(iter_fit):
            transitions = self.replay_buffer.sample(self._config["batch_size"])
            s = to_torch(transitions[:,0])
            a = to_torch(transitions[:,1])
            r = to_torch(transitions[:,2])[:,None]
            s_prime = to_torch(transitions[:,3])
            done = to_torch(transitions[:,4])[:,None]
            
            a_prime = self.policy_target.forward(s_prime)
            noise = torch.clamp(torch.randn(a_prime.shape) * self._config["policy_noise"], -self._noise_clamp, self._noise_clamp)
            a_prime += noise 
            q_prime_1 = self.Q_target_1.Q_value(observations=s_prime, actions= a_prime)
            q_prime_2 = self.Q_target_2.Q_value(observations=s_prime, actions= a_prime)
            
            q_prime = torch.min(q_prime_1, q_prime_2)
            
            gamma = self._config["discount"]
            td_target = r + gamma * (1 - done) * q_prime
            
            fit_loss_1 = self.Q_net_1.fit(observations=s, actions=a, targets=td_target)
            fit_loss_2 = self.Q_net_2.fit(observations=s, actions=a, targets=td_target)
            
            if self.train_iter % self._config["policy_update_freq"] == 0:
                self.optimizer.zero_grad()
                q = self.Q_net_1.Q_value(observations=s, actions=self.policy_net.forward(s))
                actor_loss = -q.mean()
                actor_loss.backward()
                self.optimizer.step()
                
                self._sliding_update(self.Q_target_1, self.Q_net_1)
                self._sliding_update(self.Q_target_2, self.Q_net_2)
                self._sliding_update(self.policy_target, self.policy_net)
                
                losses.append((fit_loss_1, fit_loss_2, actor_loss.item()))
            else:
                losses.append((fit_loss_1, fit_loss_2, None))
                
                
        return losses
                
            
            
            
            
        