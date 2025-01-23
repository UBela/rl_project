import numpy as np

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
