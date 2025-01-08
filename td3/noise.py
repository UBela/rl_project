import numpy as np
import torch

class OUNoise():
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.
    
    """
    def __init__(self, shape: int, theta: float = 0.15, dt: float = 1e-2):
        
        self.shape = shape
        self.theta = theta
        self.dt = dt
        self.noise_prev = np.zeros(self.shape)
        self.reset()
        
    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev + 
            self.theta * (-self.noise_prev) * self.dt + 
            np.sqrt(self.dt) * np.random.normal(size=self.shape)
        )
        self.noise_prev = noise
        return noise
    
    def reset(self):
        self.noise_prev = np.zeros(self.shape)
        

class GaussianNoise():
    """
    Gaussian noise for exploration.
    
    """
    def __init__(self, shape: int, scale: float = 0.1):
        self.shape = shape
        self.scale = scale
        self.reset()
        
    def __call__(self) -> np.ndarray:
        return np.random.normal(torch.zeros(self.shape), self.scale)
    
    def reset(self):
        pass
        