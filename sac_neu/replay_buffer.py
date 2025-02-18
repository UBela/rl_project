import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.transitions = []
        self.max_size = max_size

    def add_transition(self, transition):
        if len(self.transitions) >= self.max_size:
            self.transitions.pop(0)
        self.transitions.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.transitions), batch_size, replace=False)
        batch = [self.transitions[i] for i in indices]
        return zip(*batch)

    def __len__(self):
        return len(self.transitions)
