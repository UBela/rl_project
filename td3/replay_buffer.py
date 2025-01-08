import torch
import numpy as np

class ReplayBuffer:
    """
    A simple replay buffer to store and sample transitions.
    
    """
    def __init__(self, max_size: int =100000):
        self.transitions = np.array([])
        self.size = 0
        self.currend_idx = 0
        self. max_size = max_size
        
    def add_transition(self, new_transitions: list):
        """
        Add new transitions to the buffer.
        """
        if self.size == 0:
            blank_buffer = [np.asarray(new_transitions, dtype=object)] * self.max_size
            self.transitions = np.array(blank_buffer)
            
        self.transitions[self.currend_idx,:] = np.asarray(new_transitions, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.currend_idx + 1) % self.max_size # overwrite the oldest transitions if buffer is full
        
    def sample(self, batch_size: int = 1):
        """
        Sample a batch of transitions from the buffer.
        """
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return self.transitions[indices]
    
    
    