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
    
    def __init__(self, max_size, alpha=0.4, beta=0.4, eps=1e-5):
        self.transitions = np.array([])
        self.size = 0
        self.current_idx = 0  
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(max_size)
        self.eps = eps
        self.max_priority = 1.0
        
    
    def add_transition(self, new_transitions):
        """
        Add new transitions to the buffer.
        """
        if self.size == 0:
            blank_buffer = [np.asarray(new_transitions, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)
        
        initial_priority = self.max_priority ** self.alpha
        self.tree.add(initial_priority, data=self.current_idx)
        self.transitions[self.current_idx, :] = np.asarray(new_transitions, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size  
        
    def sample(self, batch_size: int = 1):
        
        if batch_size > self.size:
            batch_size = self.size
        segment = self.tree.total() / batch_size
       
        tree_idxs = []
        sample_idxs = []
        priorities = np.zeros(batch_size, dtype=np.float32)
        # get minimum probabilty, ie. the smallest leaf in the sum tree and maximum weight
        p_min = self.tree.tree[-self.tree.capacity:].min() / self.tree.total()
        max_weight = (self.size * p_min) ** (-self.beta)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            rand_v = np.random.uniform(a, b)
           
            leaf_idx, priority, sample_idx = self.tree.get_leaf(rand_v)
            priorities[i] = priority
            tree_idxs.append(leaf_idx)
            sample_idxs.append(sample_idx)
           
        
        probs = priorities / self.tree.total()
        
        weights = (self.size * (probs + self.eps)) ** (-self.beta)
        weights = weights / (max_weight + self.eps)

    
        #store experiences in a numpy array so that each row is one transition
    
        return self.transitions[sample_idxs, :], tree_idxs, weights
        
    def update_priorities(self, tree_idxs, errors):
        errors = errors + self.eps
        #errors = np.minimum(errors, self.max_priority)
        priorities = errors ** self.alpha
        #assert(priorities > 0.).all()
        for idx, p in zip(tree_idxs, priorities):
           
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    
    def __len__(self):
        return self.size
    
    def update_beta(self, update_per_beta):
        self.beta = min(1.0, self.beta + update_per_beta)