import numpy as np
from utils.sum_tree import SumTree
import torch
class ReplayBuffer:
    """
    A simple replay buffer to store and sample transitions.
    """
    def __init__(self, max_size: int = 100000):
        self.transitions = np.array([])
        self.size = 0
        self.current_idx = 0  
        self.max_size = max_size

    def add_transition(self, new_transitions: list):
        """
        Add new transitions to the buffer.
        """
        if self.size == 0:
            blank_buffer = [np.asarray(new_transitions, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)
        
        self.transitions[self.current_idx, :] = np.asarray(new_transitions, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size  
    
    def sample(self, batch_size: int = 1):
        """
        Sample a batch of transitions from the buffer.
        """
        if batch_size > self.size:
            batch_size = self.size
        self.inds = np.random.choice(range(self.size), size=batch_size, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        """
        Retrieve all stored transitions up to the current size.
        """
        return self.transitions[:self.size]

    # get length of buffer
    def __len__(self):
        return self.size
    
    
class PriorityReplayBuffer:
    
    def __init__(self, max_size, alpha=0.4, beta=0.4, eps=1e-5, update_per_beta=3e-5):
        self.transitions = np.array([])
        self.size = 0
        self.current_idx = 0  
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(max_size)
        self.eps = eps
        self.max_priority = 1.0
        self.update_per_beta = update_per_beta
        
    
    def add_transition(self, sample):

        priority = np.amax(self.tree.tree[-self.tree.capacity:])
        if priority == 0: priority = self.max_priority
        
        self.tree.add(priority, sample) 
        
    def sample(self, batch_size: int = 1):
        
        if batch_size > self.size:
            batch_size = self.size
       

        # Update beta every time a batch is sampled
        self.beta = min(1.0, self.beta + self.update_per_beta)
        segment = self.tree.total() / batch_size
       
        tree_idxs = []
        priorities = np.zeros(batch_size, dtype=np.float32)
        experiences =  np.zeros(batch_size, dtype=object)
        # get minimum probabilty, ie. the smallest leaf in the sum tree and maximum weight
        p_min = self.tree.tree[-self.tree.capacity:].min() / self.tree.total()
        max_weight = (batch_size * p_min) ** (-self.beta)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            rand_v = np.random.uniform(a, b)
           
            leaf_idx, priority, transitions = self.tree.get_leaf(rand_v)
            priorities[i] = priority
            tree_idxs.append(leaf_idx)
            experiences[i] = np.asarray(transitions, dtype=object)
           
        
        probs = priorities / self.tree.total()
        
        weights = (batch_size * probs) ** (-self.beta)
        weights = weights / max_weight
    
        #store experiences in a numpy array so that each row is one transition
        
        print(experiences.shape)
        print(batch_size)
        return experiences, tree_idxs, weights
        
    def update_priorities(self, tree_idxs, errors):
        errors = errors + self.eps
        errors = np.minimum(errors, self.max_priority)
        priorities = errors ** self.alpha
        assert(priorities > 0.).all()
        for idx, p in zip(tree_idxs, priorities):
           
            self.tree.update(idx, p)
            
    
    def __len__(self):
        return self.size