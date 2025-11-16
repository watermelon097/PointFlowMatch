import torch
import numpy as np


def fps_numpy(points: np.ndarray, n_samples: int):
    N, _ = points.shape
    sample_idx = np.zeros(n_samples, dtype=int)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        sample_idx[i] = farthest
        dist = np.sum((points - points[farthest]) ** 2, axis=-1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return sample_idx

def get_timesteps(schedule: str, k_steps: int, exp_scale: float = 1.0):
    t = torch.linspace(0, 1, k_steps + 1)[:-1]
    if schedule == "linear":
        dt = torch.ones(k_steps) / k_steps
    elif schedule == "cosine":
        dt = torch.cos(t * torch.pi) + 1
        dt /= torch.sum(dt)
    elif schedule == "exp":
        dt = torch.exp(-t * exp_scale)
        dt /= torch.sum(dt)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    t0 = torch.cat((torch.zeros(1), torch.cumsum(dt, dim=0)[:-1]))
    return t0, dt
