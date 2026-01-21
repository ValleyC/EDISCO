"""Schedulers for Denoising Diffusion Probabilistic Models with continuous-time support"""

import math
import numpy as np
import torch


class GaussianDiffusion(object):
    """Gaussian Diffusion process with linear beta scheduling"""
    
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T
        
        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0)  # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        
        self.betabar = np.cumprod(self.beta)
        self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
        self.alphabar = np.cumprod(self.alpha)
    
    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2
    
    def sample(self, x0, t):
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'
        
        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
        return xt, epsilon


class CategoricalDiffusion(object):
    """Categorical Diffusion process with linear beta scheduling"""
    
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T
        
        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0)  # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        
        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))
        
        self.Qs = (1 - beta) * eye + (beta / 2) * ones
        
        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)
    
    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2
    
    def sample(self, x0_onehot, t):
        # Select noise scales
        noise_dims = (x0_onehot.shape[0],) + tuple((1 for _ in x0_onehot.shape[1:-1])) + (2, 2)
        Qtbar = torch.from_numpy(self.Q_bar[t]).view(noise_dims).to(x0_onehot.device).float()
        
        # Sample from categorical distribution
        prob = (Qtbar @ x0_onehot.unsqueeze(-1)).squeeze(-1)
        xt = torch.distributions.categorical.Categorical(prob).sample()
        return xt


class InferenceSchedule(object):
    def __init__(self, inference_schedule, T=1000, inference_T=1000):
        self.T = T
        self.inference_T = inference_T
        self.inference_schedule = inference_schedule
    
    def __getitem__(self, i):
        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)
            
            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
            t1 = np.clip(t1, 1, self.T)
            
            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        else:
            raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))


# Add support for continuous-time schedules
class ContinuousTimeSchedule:
    """Schedule for continuous-time diffusion sampling"""
    
    def __init__(self, schedule_type="linear", num_steps=50):
        self.schedule_type = schedule_type
        self.num_steps = num_steps
    
    def get_timesteps(self):
        """Get timesteps for continuous-time sampling from t=1 to t=0"""
        if self.schedule_type == "linear":
            return torch.linspace(1.0, 0.0, self.num_steps + 1)[:-1]
        elif self.schedule_type == "cosine":
            # Cosine schedule for adaptive stepping
            s = torch.linspace(0, 1, self.num_steps + 1)
            timesteps = 1 - (torch.cos(s * math.pi / 2) ** 2)[:-1]
            return timesteps.flip(0)  # Reverse to go from 1 to 0
        elif self.schedule_type == "quadratic":
            t = torch.linspace(1.0, 0.0, self.num_steps + 1)[:-1]
            return t ** 2
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")