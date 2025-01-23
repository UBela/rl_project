import torch
import numpy as np
from networks import Critic_Net, Actor_Net
from replay_buffer import ReplayBuffer
import pink
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class TD3Agent(object):
    def __init__(self, observation_space, action_space, **userconfig):
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        
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
    
        self.replay_buffer = ReplayBuffer(max_size=self._config["buffer_size"])
        
        self.critic_net = Critic_Net(
            observation_dim=self._observation_space.shape[0], 
            action_dim=self._action_n, 
            hidden_dims=self._config["hidden_dim_critic"],
            lr=self._config["critic_lr"]
        )
        self.critic_net_target = Critic_Net(
            observation_dim=self._observation_space.shape[0],
            action_dim=self._action_n, 
            hidden_dims=self._config["hidden_dim_critic"],
            lr=0  # No optimizer for the target network
        )
        
        self.actor_net = Actor_Net(
            input_dim=self._observation_space.shape[0],
            hidden_dims=self._config["hidden_dim_actor"],
            output_dim=self._action_n,
            lr=self._config["actor_lr"]
        )
        self.actor_net_target = Actor_Net(
            input_dim=self._observation_space.shape[0],
            hidden_dims=self._config["hidden_dim_actor"],
            output_dim=self._action_n,
            lr=0  # No optimizer for the target network
        )
        
        self._copy_nets()
        
        self.train_iter = 0 
        self.total_steps = 0
        
    def _copy_nets(self):
        self.critic_net_target.load_state_dict(self.critic_net.state_dict())
        self.actor_net_target.load_state_dict(self.actor_net.state_dict())
        
    def get_action(self, observation):
        #print(self.total_steps)

        if self.total_steps < self._config["exploration_steps"]:
            action = self._action_space.sample()
            
        else:
            state = torch.from_numpy(observation.astype(np.float32)).to(device)
            action = self.actor_net.forward(state)
            action = action.cpu().detach().numpy()[0]
            
            # Add Gaussian noise to the action
            noise = np.random.normal(0, self._config["noise_std"], size=self._action_n)
            
            action = action + noise
            action = np.clip(action, -1, 1)
            
        self.total_steps += 1
        return action
            
    def store_transition(self, transition: tuple):
        self.replay_buffer.add_transition(transition)
        
    def state(self):
        return self.critic_net.state_dict(), self.actor_net.state_dict()
    
    def restore_state(self, state):
        self.critic_net.load_state_dict(state[0])
        self.actor_net.load_state_dict(state[1])
        self._copy_nets()
        
    def _sliding_update(self, target: torch.nn.Module, source: torch.nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self._config["tau"] * param.data + (1.0 - self._config["tau"]) * target_param.data)
            
    def train(self, iter_fit: int = 32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = [] 
        self.train_iter += 1
        
        for _ in range(iter_fit):
            transitions = self.replay_buffer.sample(batch_size=self._config["batch_size"])
            s = to_torch(np.stack(transitions[:, 0]))
            a = to_torch(np.stack(transitions[:, 1]))
            r = to_torch(np.stack(transitions[:, 2])[:, None])
            s_prime = to_torch(np.stack(transitions[:, 3]))
            done = to_torch(np.stack(transitions[:, 4])[:, None])
            
            a_prime = self.actor_net_target.forward(s_prime)
            noise = torch.clamp(torch.randn(a_prime.shape) * self._config["noise_std"], -self._noise_clamp, self._noise_clamp)
            #print("action space", self._action_space.low, self._action_space.high)
            #print("action_n", self._action_n)

            a_prime = torch.clamp(a_prime + noise, -1, 1)
            
            # Get Q-values from the target Q-function
            q1_prime, q2_prime = self.critic_net_target.forward(torch.cat([s_prime, a_prime], dim=1))
            
            q_prime_min = torch.min(q1_prime, q2_prime)
            #print("qprime min",q_prime_min)
            gamma = self._config["discount"]
            td_target = r + gamma * (1 - done) * q_prime_min
            
            # Update critic_net
            q1, q2 = self.critic_net.forward(torch.cat([s, a], dim=1))
            #q1, q2 = q1.squeeze(dim=1), q2.squeeze(dim=1)
            
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            
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



