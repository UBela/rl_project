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
    """
    A prioritized replay buffer that stores and samples transitions based on their priority, implemented using a SumTree.

    The buffer stores transitions and assigns priorities to each transition based on their TD error. 
    The priority of each transition is used to sample transitions more likely to have high TD errors, thus improving the learning efficiency. 

    The buffer uses a SumTree data structure to efficiently store and retrieve transitions based on their priorities.

    Attributes:
        max_size (int): The maximum number of transitions the buffer can hold.
        alpha (float): The exponent used to calculate the priority of a transition (higher values make the priorities more skewed).
        beta (float): The importance-sampling weight, which adjusts the bias introduced by prioritization.
        eps (float): A small value added to probabilities to avoid division by zero.
        transitions (np.ndarray): Array that stores the transitions in the buffer.
        size (int): The current number of transitions in the buffer.
        current_idx (int): The index for the next transition to be added to the buffer.
        tree (SumTree): The SumTree structure used to store priorities and efficiently sample transitions.
        max_priority (float): The maximum priority in the buffer, used to initialize new transitions with high priority.

    Methods:
        add_transition(new_transitions):
            Adds a new transition to the buffer with an initial priority based on the maximum priority.

        sample(batch_size=1):
            Samples a batch of transitions from the buffer according to their priorities, returning the transitions, tree indices, and importance-sampling weights.

        update_priorities(tree_idxs, errors):
            Updates the priorities of transitions in the buffer based on the new TD errors, which are used for further prioritization.

        __len__():
            Returns the current size of the buffer.

        update_beta(update_per_beta):
            Updates the beta value gradually, which is used for computing importance-sampling weights.
            
        This docstring was generated with the help of chatGPT
    """

    
    def __init__(self, max_size, alpha=0.6, beta=0.4, eps=0.01):
        self.transitions = np.array([])
        self.size = 0
        self.current_idx = 0  
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(max_size)
        self.eps = eps
        self.max_priority = 1.0
        assert self.max_size > 0 and self.max_size & (self.max_size - 1) == 0, "Max size must be a power of 2"
        
    
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
        
        leafs = self.tree.tree[-self.tree.capacity:]
        # filter out zeroes, in case the tree is not full, avoided by prefilling replay buffer
        non_zero_leafs = leafs[leafs != 0]
       
        smallest_leaf = non_zero_leafs.min()
        p_min = smallest_leaf / self.tree.total()
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
        
        #avoid division by zero when the tree is not full, avoided by prefilling replay buffer
        if (probs == 0).any():
            probs = probs + 1e-5
        weights = (self.size * probs) ** (-self.beta)
    
        weights = weights / max_weight
        
        return self.transitions[sample_idxs, :], tree_idxs, weights
        
    def update_priorities(self, tree_idxs, errors):
        errors = errors + self.eps
        priorities = errors ** self.alpha
        for idx, p in zip(tree_idxs, priorities):
           
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    
    def __len__(self):
        return self.size
    
    def update_beta(self, update_per_beta):
        self.beta = min(1.0, self.beta + update_per_beta)