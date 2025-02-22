import numpy as np

class ReplayBuffer:
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = []
        self.current_idx = 0  

    def add_transition(self, transition):
        """ Speichert eine neue Transition """
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.current_idx] = transition
        
        self.current_idx = (self.current_idx + 1) % self.max_size  

    def sample(self, batch_size: int = 1):
        """ Nimmt eine zufällige Stichprobe """
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)
