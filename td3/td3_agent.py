import torch
import numpy as np
from td3.networks import Critic_Net, Actor_Net
from utils.replay_buffer import ReplayBuffer, PriorityReplayBuffer
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

        
class TD3Agent(object):
    def __init__(self, observation_space, action_space, device, userconfig):
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.shape[0] //2 # agent should output action of shape (4,)
        self._config = {
            "actor_lr": 0.0003,
            "critic_lr": 0.0003,
            "discount": 0.999,
            "tau": 0.005,
            "batch_size": 128,
            "hidden_dim_critic": [256, 256],
            "hidden_dim_actor": [256, 256],
            "noise": "Gaussian",
            "noise_clip": 0.3,
            "noise_std": 0.2,
            "policy_update_freq": 2,
            "buffer_size": int(1e6),
            "exploration_steps": 2000,
            
        }
        
        self._config.update(userconfig)    
        self._noise_clamp = self._config["noise_clip"]
        self._use_prioritized = self._config["use_PER"]
        if not self._use_prioritized:
            print("Use simple Experience Replay")
            self.replay_buffer = ReplayBuffer(max_size=self._config["buffer_size"])
        else: 
            print("Use Prioritized Experience Replay")
            self.replay_buffer = PriorityReplayBuffer(max_size=self._config["buffer_size"], 
                                                    alpha=self._config["per_alpha"], 
                                                    beta=self._config["per_beta"])
        self.critic_net = Critic_Net(
            observation_dim=self._observation_space.shape[0], 
            action_dim=self._action_n, 
            hidden_dims=self._config["hidden_dim_critic"],
            lr=self._config["critic_lr"]
        ).to(device)
        self.critic_net_target = Critic_Net(
            observation_dim=self._observation_space.shape[0],
            action_dim=self._action_n, 
            hidden_dims=self._config["hidden_dim_critic"],
            lr=0  # No optimizer for the target network
        ).to(device)
        
        self.actor_net = Actor_Net(
            input_dim=self._observation_space.shape[0],
            hidden_dims=self._config["hidden_dim_actor"],
            output_dim=self._action_n,
            lr=self._config["actor_lr"]
        ).to(device)
        self.actor_net_target = Actor_Net(
            input_dim=self._observation_space.shape[0],
            hidden_dims=self._config["hidden_dim_actor"],
            output_dim=self._action_n,
            lr=0  # No optimizer for the target network
        ).to(device)
        
        self._copy_nets()
        
        self.train_iter = 0 
        self.total_steps = 0
        

    def _copy_nets(self):
        self.critic_net_target.load_state_dict(self.critic_net.state_dict())
        self.actor_net_target.load_state_dict(self.actor_net.state_dict())
        
    def act(self, observation):
        #print(self.total_steps)
        if self.total_steps < self._config["exploration_steps"]:
            # random action of shape (self._action_n,) bounded by low and high
            action = np.random.uniform(self._action_space.low[0], self._action_space.high[0], size=self._action_n)
        else:
            with torch.no_grad():
                state = torch.from_numpy(observation.astype(np.float32)).to(device)
                action = self.actor_net.forward(state)
                action = action.cpu().detach().numpy()
                # Add Gaussian noise to the action
                noise = np.random.normal(0, self._config["noise_std"], size=self._action_n)
                action = action + noise
                action = np.clip(action, self._action_space.low[0], self._action_space.high[0])
            
        self.total_steps += 1
        return action
            
    def store_transition(self, transition: tuple):
        
        self.replay_buffer.add_transition(transition)
    
    def update_per_beta(self, update_per_beta):
        self.replay_buffer.update_beta(update_per_beta)
        
    def state(self):
        return self.critic_net.state_dict(), self.actor_net.state_dict()
    
    def restore_state(self, state):
        self.critic_net.load_state_dict(state[0])
        self.actor_net.load_state_dict(state[1])
        self._copy_nets()
    
    def set_to_train(self):
        self.actor_net.train()
        self.critic_net.train()
        self.actor_net_target.train()
        self.critic_net_target.train()
        
    def set_to_eval(self):
        self.actor_net.eval()
        self.critic_net.eval()
        self.actor_net_target.eval()
        self.critic_net_target.eval()
    
    
        
    def mse(self, pred, target, weight):
        """MSE with importance sampling weights from PER.

        Args:
            pred (_type_): _description_
            target (_type_): _description_
            weight (_type_): _description_
        """
        td_error = pred - target
        
        weighted_squared_error = weight * td_error * td_error
        return weighted_squared_error.mean()
    
    
    def _sliding_update(self, target: torch.nn.Module, source: torch.nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self._config["tau"] * param.data + (1.0 - self._config["tau"]) * target_param.data)
            
    def train(self, iter_fit: int = 32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = [] 
        self.train_iter += 1
        
        for _ in range(iter_fit):
            
            if not self._use_prioritized:
                transitions = self.replay_buffer.sample(batch_size=self._config["batch_size"])
            else:
                transitions, idxs, weights = self.replay_buffer.sample(batch_size=self._config["batch_size"])
                weights = torch.from_numpy(weights).to(device)
                
            s = to_torch(np.stack(transitions[:, 0])).to(device)
            a = to_torch(np.stack(transitions[:, 1])).to(device)
            r = to_torch(np.stack(transitions[:, 2])[:, None]).to(device)
            s_prime = to_torch(np.stack(transitions[:, 3])).to(device)
            done = to_torch(np.stack(transitions[:, 4])[:, None]).to(device)
            
            with torch.no_grad():
                
                a_prime = self.actor_net_target.forward(s_prime).to(device)
                noise = torch.clamp(torch.randn(a_prime.shape) * self._config["noise_std"], -self._noise_clamp, self._noise_clamp).to(device)
                a_prime = torch.clamp(a_prime + noise, -1, 1)
                
                # Get Q-values from the target Q-function
                q1_prime, q2_prime = self.critic_net_target.forward(torch.cat([s_prime, a_prime], dim=1))
                
                q_prime_min = torch.min(q1_prime, q2_prime).to(device)
                gamma = self._config["discount"]
                td_target = r + gamma * (1 - done) * q_prime_min
                td_target = td_target.to(device)
                
            # Update critic_net
            q1, q2 = self.critic_net.forward(torch.cat([s, a], dim=1))
            q1, q2 = q1.to(device), q2.to(device)
            
            if not self._use_prioritized:
                critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            else:
                critic_loss = self.mse(q1, td_target, weights) + self.mse(q2, td_target, weights)
            
            td_error = torch.abs(q1 - td_target).detach().cpu().numpy()
            
            #check if q1 or td_target is nan
            if torch.isnan(q1).any() or torch.isnan(td_target).any():
                print("weights",weights)
                print("loss",critic_loss)
                print("q1: ", q1)
                print("td_target: ", td_target)
                raise ValueError("NaN detected in TD3 loss computation!")
                
             
            if self._use_prioritized:
                self.replay_buffer.update_priorities(idxs, td_error)
            
            self.critic_net.optimizer.zero_grad()
            critic_loss.backward()
            self.critic_net.optimizer.step()
           
            
            # Policy update
            if self.train_iter % self._config["policy_update_freq"] == 0:
                
                q1 = self.critic_net.Q_value(s, self.actor_net.forward(s))
                actor_loss = -q1.mean()
                self.actor_net.optimizer.zero_grad()
                actor_loss.backward()
                self.actor_net.optimizer.step()
                
                # Update target networks
                self._sliding_update(self.critic_net_target, self.critic_net)
                self._sliding_update(self.actor_net_target, self.actor_net)
                
                losses.append((critic_loss.item(), actor_loss.item()))
            else:
                losses.append((critic_loss.item(), None))
                
        return losses



